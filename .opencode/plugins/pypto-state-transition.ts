import { type Plugin, tool } from "@opencode-ai/plugin";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import {
  applyTransition,
  migrateLegacyState,
  type OrchestratorState,
  type TransitionAction,
} from "./lib/state-transition-core";

type GateFinding = {
  rule_id: string;
  severity: string;
  status: string;
  message: string;
  file: string;
};

type GateSummary = {
  warnCount: number;
  infoCount: number;
  /** FAIL findings included when gate blocks — gives the agent actionable detail. */
  failFindings: GateFinding[];
};

const ALLOWED_ACTIONS = new Set<TransitionAction>([
  "init",
  "start_stage",
  "complete_stage",
  "fail_stage",
  // Stage 6 三者循环（coder/verifier/debugger）状态管理 actions
  "advance_module",
  "record_gate",
  "record_module_attempt",
]);

/** 仅允许以下 agent 调用 state_transition 工具 */
const ALLOWED_AGENTS = new Set<string>([
  "pypto-op-orchestrator",
]);

function parseState(content: string): OrchestratorState {
  const parsed = JSON.parse(content);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("invalid orchestrator state file content");
  }
  return parsed as OrchestratorState;
}

function parseGateSummary(raw: string): GateSummary {
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const summary = parsed.summary;
    if (!summary || typeof summary !== "object" || Array.isArray(summary)) {
      return { warnCount: 0, infoCount: 0, failFindings: [] };
    }
    const warnCount = Number((summary as Record<string, unknown>).warn);
    const infoCount = Number((summary as Record<string, unknown>).info);

    // Extract FAIL findings for detailed error reporting
    const failFindings: GateFinding[] = [];
    const findings = parsed.findings;
    if (Array.isArray(findings)) {
      for (const f of findings) {
        if (f && typeof f === "object" && (f as Record<string, unknown>).status === "FAIL") {
          failFindings.push({
            rule_id: String((f as Record<string, unknown>).rule_id ?? ""),
            severity: String((f as Record<string, unknown>).severity ?? ""),
            status: "FAIL",
            message: String((f as Record<string, unknown>).message ?? ""),
            file: String((f as Record<string, unknown>).file ?? ""),
          });
        }
      }
    }

    return {
      warnCount: Number.isFinite(warnCount) ? warnCount : 0,
      infoCount: Number.isFinite(infoCount) ? infoCount : 0,
      failFindings,
    };
  } catch {
    return { warnCount: 0, infoCount: 0, failFindings: [] };
  }
}

function buildInitialState(opDir: string): OrchestratorState {
  return {
    operator_name: path.basename(opDir),
    current_stage: 1,
    stage_status: {
      "1": "pending",
      "2": "pending",
      "3": "pending",
      "4": "pending",
      "5": "pending",
      "6": "pending",
      "7": "pending",
    },
    stage_retry_count: {
      "1": 0,
      "2": 0,
      "3": 0,
      "4": 0,
      "5": 0,
      "6": 0,
      "7": 0,
    },
    // Stage 6 三者循环（coder/verifier/debugger）相关字段
    gate_status: {
      "0": "pending",
      "1": "pending",
      "2": "pending",
      "3": "pending",
      "4": "pending",
    },
    current_subphase: null,
    last_failure: null,
    // module_state 在第一次 advance_module(stage=6, k=1, total_modules=N) 时由 core 创建
    perf_iteration: {
      count: 0,
      last_improvement: 0,
      consecutive_no_improvement: 0,
    },
    last_updated: new Date().toISOString(),
  };
}

function readStateOrInit(statePath: string): OrchestratorState {
  if (!fs.existsSync(statePath)) {
    return buildInitialState(path.dirname(statePath));
  }
  const raw = fs.readFileSync(statePath, "utf8");
  // 旧格式状态文件（无 gate_status / current_subphase / last_failure）需要迁移补默认值。
  return migrateLegacyState(parseState(raw));
}

function writeStateAtomically(statePath: string, state: OrchestratorState): void {
  const tmpPath = `${statePath}.tmp`;
  fs.writeFileSync(tmpPath, `${JSON.stringify(state, null, 2)}\n`, "utf8");
  fs.renameSync(tmpPath, statePath);
}

/** Stage 1 完成时记录 SPEC.md 内容 hash，后续阶段用于冻结校验。 */
const SPEC_HASH_KEY = "spec_md_hash";

function computeFileHash(filePath: string): string | null {
  if (!fs.existsSync(filePath)) return null;
  const content = fs.readFileSync(filePath, "utf8");
  return crypto.createHash("sha256").update(content).digest("hex");
}

export const PyptoStateTransitionPlugin: Plugin = async (input) => {
  const $ = input.$;
  const client = input.client;
  const baseDir = input.worktree || input.directory || process.cwd();
  const lintScript = new URL(
    "../../.agents/hooks/pypto-op-lint/pypto_op_lint.py",
    import.meta.url,
  ).pathname;

  async function runGateIfNeeded(opDir: string, action: TransitionAction, stage: number): Promise<GateSummary> {
    if (action !== "complete_stage") return { warnCount: 0, infoCount: 0, failFindings: [] };

    // Use nothrow() so stdout is available even when exit code is non-zero.
    // The lint script writes findings JSON to stdout and exits 2 on S1 FAIL.
    const result = await $`python3 ${lintScript} --check-gate --op-dir ${opDir} --stage ${stage}`
      .cwd(baseDir).quiet().nothrow();
    const raw = result.stdout.toString();
    const summary = parseGateSummary(raw);

    if (result.exitCode !== 0) {
      // Build detailed error lines from FAIL findings
      const details = summary.failFindings
        .map((f) => `  [${f.rule_id}][${f.severity}] ${f.message}${f.file ? ` (${f.file})` : ""}`)
        .join("\n");
      throw new Error(
        details
          ? `以下规则违规导致门禁阻断：\n${details}`
          : `lint 脚本返回 exit code ${result.exitCode}，但未解析到具体违规项`,
      );
    }

    return summary;
  }

  return {
    tool: {
      state_transition: tool({
        description:
          "Safely transition .orchestrator_state.json with stage gate enforcement. " +
          "Stage-level actions: init (stage=1 only, first call), start_stage (set stage to in_progress — for init or retry after failure), " +
          "complete_stage (gate check + mark done + auto-advance to next stage), fail_stage (mark failed + increment retry). " +
          "Stage 6 module-loop actions: advance_module (stage=6, module_index=k, total_modules on first call) — advances active_module and marks the previous one verified; " +
          "record_gate (gate=0..4, gate_status=pending|in_progress|passed|failed) — updates GATE 0..4 status; " +
          "record_module_attempt (stage=6, module_index=k, failure_category?) — increments module_attempts[k] and stores last_failure on category. " +
          "Any action may also pass an optional `subphase` to update current_subphase (coding|verifying|debugging|scaffolding|e2e_gate|phase_d|null).",
        args: {
          opDir: tool.schema.string(),
          action: tool.schema.string(),
          stage: tool.schema.number(),
          reason: tool.schema.string().optional(),
          // Stage 6 module-loop parameters
          module_index: tool.schema.number().optional(),
          total_modules: tool.schema.number().optional(),
          gate: tool.schema.number().optional(),
          gate_status: tool.schema.string().optional(),
          failure_category: tool.schema.string().optional(),
          evaluation_report_path: tool.schema.string().optional(),
          subphase: tool.schema.string().optional(),
        },
        execute: async (args, context) => {
          // ── Agent 权限校验：仅 pypto-op-orchestrator 可调用 ──
          const callerAgent = context?.agent ?? "";
          if (!ALLOWED_AGENTS.has(callerAgent)) {
            throw new Error(
              `permission denied: state_transition is restricted to pypto-op-orchestrator, ` +
              `but was called by "${callerAgent || "(unknown)"}". ` +
              `Subagents must not modify .orchestrator_state.json; return stage results to the orchestrator instead.`,
            );
          }

          const action = String(args.action) as TransitionAction;
          if (!ALLOWED_ACTIONS.has(action)) {
            throw new Error(`unsupported action: ${args.action}`);
          }

          const opDir = path.isAbsolute(args.opDir)
            ? args.opDir
            : path.resolve(baseDir, args.opDir);
          const statePath = path.join(opDir, ".orchestrator_state.json");
          const prevState = readStateOrInit(statePath);

          let gateSummary: GateSummary = { warnCount: 0, infoCount: 0, failFindings: [] };
          try {
            gateSummary = await runGateIfNeeded(opDir, action, args.stage);
          } catch (error) {
            const detail = error instanceof Error ? error.message : String(error);
            throw new Error(
              `[pypto-op-lint] 交付门禁阻断（ERROR 级违规）：action=${action}, stage=${args.stage}, op_dir=${opDir}\n${detail}`,
            );
          }

          // ── SPEC.md 冻结校验 ──
          // complete_stage(1) 时记录 SPEC.md hash；
          // complete_stage(N>=3) 时校验 hash 不变，防止 subagent 绕过 hook 修改 SPEC。
          const specPath = path.join(opDir, "SPEC.md");
          if (action === "complete_stage" && args.stage === 1) {
            const hash = computeFileHash(specPath);
            if (hash) {
              prevState[SPEC_HASH_KEY] = hash;
            }
          }
          if (action === "complete_stage" && args.stage >= 3) {
            const savedHash = prevState[SPEC_HASH_KEY];
            if (typeof savedHash === "string") {
              const currentHash = computeFileHash(specPath);
              if (currentHash && currentHash !== savedHash) {
                throw new Error(
                  `[pypto-op-lint] SPEC.md 冻结违规：SPEC.md 在 Stage 2 完成后被修改。` +
                  `当前 hash=${currentHash.slice(0, 12)}… 与记录 hash=${savedHash.slice(0, 12)}… 不一致。` +
                  `如需变更需求规格，应通过 fail_stage 回退到 Stage 1 重新审核。`,
                );
              }
            }
          }

          // Stage 6 三者循环参数透传到 core；非相关 action 时这些字段被忽略。
          const nextState = applyTransition(prevState, {
            action,
            stage: args.stage,
            module_index: typeof args.module_index === "number" ? args.module_index : undefined,
            total_modules: typeof args.total_modules === "number" ? args.total_modules : undefined,
            gate: typeof args.gate === "number" ? args.gate : undefined,
            gate_status:
              typeof args.gate_status === "string"
                ? (args.gate_status as "pending" | "in_progress" | "passed" | "failed")
                : undefined,
            failure_category:
              typeof args.failure_category === "string" ? args.failure_category : undefined,
            evaluation_report_path:
              typeof args.evaluation_report_path === "string" ? args.evaluation_report_path : undefined,
            subphase:
              typeof args.subphase === "string"
                ? (args.subphase as
                    | "coding"
                    | "verifying"
                    | "debugging"
                    | "scaffolding"
                    | "e2e_gate"
                    | "phase_d")
                : undefined,
          });

          // 将 SPEC hash 持久化到 state（init/complete_stage(1) 时写入）
          if (prevState[SPEC_HASH_KEY]) {
            nextState[SPEC_HASH_KEY] = prevState[SPEC_HASH_KEY];
          }

          writeStateAtomically(statePath, nextState);

          if (gateSummary.warnCount > 0 || gateSummary.infoCount > 0) {
            // 记录日志失败不应影响状态迁移结果（避免“落盘成功但工具返回异常”）。
            try {
              await client.app.log({
                body: {
                  service: "pypto-state-transition",
                  level: "info",
                  message: `[state_transition] ${action} stage=${args.stage} completed`,
                  extra: {
                    opDir,
                    reason: args.reason ?? "",
                    warnCount: gateSummary.warnCount,
                    infoCount: gateSummary.infoCount,
                  },
                },
              });
            } catch {
              // no-op
            }
          }

          return JSON.stringify({
            ok: true,
            action,
            stage: args.stage,
            next_stage: nextState.current_stage,
            statePath,
            warnCount: gateSummary.warnCount,
            infoCount: gateSummary.infoCount,
            // Stage 6 三者循环相关字段（非 Stage 6 时为 undefined / null）
            module_state: nextState.module_state,
            gate_status: nextState.gate_status,
            current_subphase: nextState.current_subphase,
            last_failure: nextState.last_failure,
          });
        },
      }),
    },
  };
};
