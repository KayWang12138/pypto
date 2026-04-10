import type { Plugin } from "@opencode-ai/plugin";
import fs from "node:fs";
import path from "node:path";

type HookOutput = {
  hookSpecificOutput?: {
    additionalContext?: string;
  };
};

const DELIVERY_GATE_REMINDER =
  "[pypto-op-lint] 若准备结束当前阶段，请在算子目录显式运行 `python3 .agents/hooks/pypto-op-lint/pypto_op_lint.py --check-gate --op-dir <算子目录> --stage <当前阶段>` 做交付自检。";

function appendMessage(output: { output?: string }, message: string): void {
  output.output = `${output.output ?? ""}\n\n${message}`.trim();
}

function formatPluginError(scope: string, error: unknown): string {
  const detail = error instanceof Error ? error.message : String(error);
  return `[pypto-op-lint plugin-error] ${scope} 自动检查失败：${detail}`;
}

function extractAdditionalContext(raw: string): string {
  if (!raw.trim()) return "";

  try {
    const parsed = JSON.parse(raw) as HookOutput;
    return parsed.hookSpecificOutput?.additionalContext ?? "";
  } catch (error) {
    return formatPluginError("post-edit 输出解析", error);
  }
}

function resolvePath(baseDir: string, filePath: string): string {
  if (!filePath) return "";
  return path.isAbsolute(filePath) ? filePath : path.resolve(baseDir, filePath);
}

function parseJsonObject(content: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(content);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    // ignore
  }
  return null;
}

function parseLintReport(content: string): {
  errorFailCount: number;
  errorFailLines: string[];
  nonErrorFailCount: number;
  nonErrorFailLines: string[];
  warnCount: number;
  warnLines: string[];
  infoCount: number;
  infoLines: string[];
} | null {
  const obj = parseJsonObject(content);
  if (!obj) return null;

  const summary = obj.summary;
  const findings = obj.findings;
  const errorFailLines: string[] = [];
  const nonErrorFailLines: string[] = [];
  const warnLines: string[] = [];
  const infoLines: string[] = [];
  if (Array.isArray(findings)) {
    for (const item of findings) {
      if (!item || typeof item !== "object" || Array.isArray(item)) continue;
      const row = item as Record<string, unknown>;
      const ruleId = String(row.rule_id ?? "UNKNOWN");
      const severity = String(row.severity ?? "S?");
      const message = String(row.message ?? "lint issue");
      if (row.status === "FAIL") {
        if (severity === "S1") {
          errorFailLines.push(`[${ruleId}][${severity}] ${message}`);
        } else {
          nonErrorFailLines.push(`[${ruleId}][${severity}] ${message}`);
        }
      }
      if (row.status === "WARN") {
        warnLines.push(`[${ruleId}][${severity}] ${message}`);
      }
      if (row.status === "INFO") {
        infoLines.push(`[${ruleId}][${severity}] ${message}`);
      }
    }
  }

  const warnCountFromSummary =
    summary && typeof summary === "object" && !Array.isArray(summary)
      ? Number((summary as Record<string, unknown>).warn)
      : NaN;
  const warnCount = Number.isFinite(warnCountFromSummary)
    ? warnCountFromSummary
    : warnLines.length;

  const infoCountFromSummary =
    summary && typeof summary === "object" && !Array.isArray(summary)
      ? Number((summary as Record<string, unknown>).info)
      : NaN;
  const infoCount = Number.isFinite(infoCountFromSummary)
    ? infoCountFromSummary
    : infoLines.length;

  return {
    errorFailCount: errorFailLines.length,
    errorFailLines,
    nonErrorFailCount: nonErrorFailLines.length,
    nonErrorFailLines,
    warnCount,
    warnLines,
    infoCount,
    infoLines,
  };
}

function inferGateStage(state: Record<string, unknown>): number | null {
  const currentStage = Number(state.current_stage);
  if (!Number.isFinite(currentStage) || currentStage < 1) return null;

  const stageStatus = state.stage_status;
  if (!stageStatus || typeof stageStatus !== "object" || Array.isArray(stageStatus)) {
    return currentStage;
  }

  const status = (stageStatus as Record<string, unknown>)[String(currentStage)];
  if (status === "completed") return currentStage;
  if (currentStage > 1) return currentStage - 1;
  return currentStage;
}

function tail(text: string, lines = 40): string {
  const split = text.split(/\r?\n/).filter((line) => line.length > 0);
  return split.slice(-lines).join("\n");
}

export const PyptoOpLintPlugin: Plugin = async (input) => {
  const $ = input.$;
  const client = input.client;
  const baseDir = input.worktree || input.directory || process.cwd();
  const lintScript = new URL(
    "../../.agents/hooks/pypto-op-lint/pypto_op_lint.py",
    import.meta.url,
  ).pathname;

  async function execHookJson(hook: string, payload: unknown): Promise<string> {
    return await $`python3 ${lintScript} --hook ${hook}`
      .env({
        PYPTO_OP_LINT_HOOK_INPUT: JSON.stringify(payload),
      })
      .quiet()
      .text();
  }

  async function runGateCheck(opDir: string, stage: number): Promise<{
    warnCount: number;
    warnLines: string[];
    infoCount: number;
    infoLines: string[];
  }> {
    const cmd = $`python3 ${lintScript} --check-gate --op-dir ${opDir} --stage ${stage}`.cwd(baseDir).quiet();

    try {
      const raw = await cmd.text();
      const report = parseLintReport(raw);
      if (report && report.errorFailCount > 0) {
        const lines = report.errorFailLines.join("\n") || `error_fail_count=${report.errorFailCount}`;
        throw new Error(
          `[pypto-op-lint] 交付门禁阻断（ERROR 级违规）：stage=${stage}, op_dir=${opDir}\n${lines}`,
        );
      }
      if (report) {
        return {
          warnCount: report.warnCount + report.nonErrorFailCount,
          warnLines: [...report.warnLines, ...report.nonErrorFailLines],
          infoCount: report.infoCount,
          infoLines: report.infoLines,
        };
      }
      return { warnCount: 0, warnLines: [], infoCount: 0, infoLines: [] };
    } catch (error) {
      const e = error as {
        stdout?: string;
        stderr?: string;
        message?: string;
      };
      const report = parseLintReport(`${e.stdout ?? ""}`) || parseLintReport(`${e.stderr ?? ""}`);
      const failLines = report && report.errorFailCount > 0
        ? report.errorFailLines.join("\n") || `error_fail_count=${report.errorFailCount}`
        : "";
      const details = failLines || tail(`${e.stdout ?? ""}\n${e.stderr ?? ""}`) || e.message || "unknown error";
      throw new Error(
        `[pypto-op-lint] 交付门禁阻断（ERROR 级违规）：stage=${stage}, op_dir=${opDir}\n${details}`,
      );
    }
  }

  return {
    "tool.execute.after": async ({ tool, args }, output) => {
      const filePath: string = args?.file_path ?? args?.path ?? "";
      if (
        filePath.endsWith("_impl.py") ||
        filePath.endsWith("_golden.py") ||
        /test_\w+\.py$/.test(filePath)
      ) {
        try {
          const raw = await execHookJson("post-edit", {
            tool_input: { file_path: filePath },
          });
          const context = extractAdditionalContext(raw);
          if (context) appendMessage(output, context);
        } catch (error) {
          appendMessage(output, formatPluginError("post-edit", error));
        }
        return;
      }

      if (
        (tool === "bash" || tool === "shell") &&
        /python3?\s+.*test_\w+\.py/.test(args?.command ?? "")
      ) {
        try {
          const currentOutput = output as {
            output?: string;
            stderr?: string;
            exit_code?: number;
            code?: number;
          };
          const raw = await execHookJson("post-bash", {
            tool_input: { command: args?.command ?? "" },
            tool_result: {
              stdout: currentOutput.output ?? "",
              stderr: currentOutput.stderr ?? "",
              exit_code:
                typeof currentOutput.exit_code === "number"
                  ? currentOutput.exit_code
                  : typeof currentOutput.code === "number"
                    ? currentOutput.code
                    : 0,
            },
          });
          const context = extractAdditionalContext(raw);
          if (context) appendMessage(output, context);
          appendMessage(output, DELIVERY_GATE_REMINDER);
        } catch (error) {
          appendMessage(output, formatPluginError("post-bash", error));
        }
      }

      return;
    },

    "tool.execute.before": async ({ tool }, output) => {
      if (tool !== "write" && tool !== "edit" && tool !== "multiedit") return;

      const filePath = output.args?.file_path ?? output.args?.path ?? "";
      const absPath = resolvePath(baseDir, filePath);
      if (!absPath) return;

      // 强制门禁：拦截 .orchestrator_state.json 的阶段推进写入
      if (absPath.endsWith(`${path.sep}.orchestrator_state.json`)) {
        const opDir = path.dirname(absPath);

        // 启动期豁免：首次创建状态文件时，必须先落盘再进行门禁校验。
        // 否则会被 OL24（状态文件不存在）提前拦截，形成“先有鸡还是先有蛋”问题。
        if (!fs.existsSync(absPath)) {
          return;
        }

        let stage: number | null = null;
        const nextContent = output.args?.content;
        if (typeof nextContent === "string") {
          const state = parseJsonObject(nextContent);
          if (state) stage = inferGateStage(state);
        }

        if (stage === null) {
          try {
            const current = fs.readFileSync(absPath, "utf8");
            const state = parseJsonObject(current);
            if (state) stage = inferGateStage(state);
          } catch {
            // ignore and fall through to default
          }
        }

        const gate = await runGateCheck(opDir, stage ?? 5);
        if (gate.warnCount > 0) {
          await client.app.log({
            body: {
              service: "pypto-op-lint",
              level: "warn",
              message: `[pypto-op-lint][WARNING] 警告：stage=${stage ?? 5}, op_dir=${opDir}`,
              extra: {
                warnCount: gate.warnCount,
                warnings: gate.warnLines,
              },
            },
          });
        }
        if (gate.infoCount > 0) {
          await client.app.log({
            body: {
              service: "pypto-op-lint",
              level: "info",
              message: `[pypto-op-lint][INFO] 建议：stage=${stage ?? 5}, op_dir=${opDir}`,
              extra: {
                infoCount: gate.infoCount,
                infos: gate.infoLines,
              },
            },
          });
        }
      }

      if (!absPath.endsWith("_impl.py")) return;

      // pre-edit: 纯 lint 检查（无副作用）
      try {
        await execHookJson("pre-edit", {
          tool_input: { file_path: absPath },
        });
      } catch (error) {
        await client.app.log({
          body: {
            service: "pypto-op-lint",
            level: "warn",
            message: formatPluginError("pre-edit", error),
            extra: { tool, filePath },
          },
        });
      }

      // pre-edit-backup: Stage 6 git 备份（独立于 lint）
      try {
        await execHookJson("pre-edit-backup", {
          tool_input: { file_path: absPath },
        });
      } catch (error) {
        await client.app.log({
          body: {
            service: "pypto-op-lint",
            level: "warn",
            message: formatPluginError("pre-edit-backup", error),
            extra: { tool, filePath },
          },
        });
      }
    },
  };
};
