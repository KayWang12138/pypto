export type TransitionAction =
  | "start_stage"
  | "complete_stage"
  | "fail_stage"
  | "init"
  | "advance_module"
  | "record_gate"
  | "record_module_attempt";

export type GateStatusValue = "pending" | "in_progress" | "passed" | "failed";

export type ModuleStatusValue = "pending" | "in_progress" | "verified" | "blocked";

export type SubphaseValue =
  | "coding"
  | "verifying"
  | "debugging"
  | "scaffolding"
  | "e2e_gate"
  | "phase_d"
  | null;

export type ModuleState = {
  N: number;
  active_module: number;
  modules_pypto_verified: number[];
  module_attempts: Record<string, number>;
  module_status: Record<string, ModuleStatusValue>;
};

export type LastFailure = {
  module: number;
  category: string;
  evaluation_report_path?: string;
};

export type TransitionInput = {
  action: TransitionAction;
  stage: number;
  /** Required for advance_module / record_module_attempt. */
  module_index?: number;
  /** Required for record_gate (0..4). */
  gate?: number;
  /** Required for record_gate. */
  gate_status?: GateStatusValue;
  /** Optional for record_module_attempt. */
  failure_category?: string;
  /** Optional for record_module_attempt — sets last_failure.evaluation_report_path. */
  evaluation_report_path?: string;
  /** Optional. Allowed during the first advance_module to seed module_state.N. */
  total_modules?: number;
  /** Optional subphase update (coding/verifying/debugging/scaffolding/e2e_gate/phase_d). */
  subphase?: SubphaseValue;
};

export type OrchestratorState = {
  operator_name?: string;
  current_stage: number;
  stage_status: Record<string, string>;
  stage_retry_count?: Record<string, number>;
  last_updated?: string;
  /** Per-module progress tracking (Stage 6 three-agent loop). */
  module_state?: ModuleState;
  /** GATE 0..4 status; key is the gate number as a string. */
  gate_status?: Record<string, GateStatusValue>;
  /** Current subphase inside Stage 6 (or null when not in Stage 6). */
  current_subphase?: SubphaseValue;
  /** Most recent verifier failure detail. Reset on PASS. */
  last_failure?: LastFailure | null;
  [key: string]: unknown;
};

const PER_MODULE_ATTEMPT_CAP = 3;
const GATE_RANGE = new Set([0, 1, 2, 3, 4]);
const GATE_STATUS_VALUES: ReadonlySet<GateStatusValue> = new Set([
  "pending",
  "in_progress",
  "passed",
  "failed",
]);
const SUBPHASE_VALUES: ReadonlySet<Exclude<SubphaseValue, null>> = new Set([
  "coding",
  "verifying",
  "debugging",
  "scaffolding",
  "e2e_gate",
  "phase_d",
]);

function cloneState(prev: OrchestratorState): OrchestratorState {
  return JSON.parse(JSON.stringify(prev));
}

function ensureNumber(val: unknown): number {
  const n = Number(val);
  if (!Number.isFinite(n) || n < 1) {
    throw new Error(`invalid stage: ${val}`);
  }
  return Math.floor(n);
}

export function applyTransition(
  prev: OrchestratorState,
  input: TransitionInput,
): OrchestratorState {
  const stage = ensureNumber(input.stage);
  const next = cloneState(prev);
  const statusMap = next.stage_status ?? {};
  next.stage_status = statusMap;
  const retryMap = next.stage_retry_count ?? {};
  next.stage_retry_count = retryMap;

  switch (input.action) {
    case "init": {
      if (stage !== 1) {
        throw new Error(`init action must target stage 1, got stage ${stage}`);
      }
      const hasInProgress = Object.values(statusMap).some((s) => s === "in_progress");
      if (hasInProgress) {
        throw new Error(`cannot init: a stage is already in_progress`);
      }
      next.current_stage = stage;
      statusMap[String(stage)] = "in_progress";
      break;
    }

    case "start_stage": {
      const otherInProgress = Object.entries(statusMap).some(
        ([k, s]) => s === "in_progress" && k !== String(stage),
      );
      if (otherInProgress) {
        throw new Error(`cannot start stage ${stage}: another stage is already in_progress`);
      }
      const key = String(stage);
      if (statusMap[key] === "completed") {
        throw new Error(`cannot start stage ${stage}: already completed`);
      }
      if (stage > 1) {
        const prevKey = String(stage - 1);
        const prevStatus = statusMap[prevKey];
        if (prevStatus !== "completed") {
          throw new Error(
            `cannot start stage ${stage}: previous stage ${stage - 1} is "${prevStatus ?? "unknown"}", not "completed"`,
          );
        }
      }
      next.current_stage = stage;
      statusMap[String(stage)] = "in_progress";
      break;
    }

    case "complete_stage": {
      if (stage !== prev.current_stage) {
        throw new Error(
          `cannot complete stage ${stage}: current_stage is ${prev.current_stage}`,
        );
      }
      const compKey = String(stage);
      if (statusMap[compKey] !== "in_progress") {
        throw new Error(
          `cannot complete stage ${stage}: status is "${statusMap[compKey]}", not "in_progress"`,
        );
      }
      statusMap[compKey] = "completed";
      // Auto-advance to next stage
      const nextStage = stage + 1;
      const nextKey = String(nextStage);
      if (nextKey in statusMap) {
        next.current_stage = nextStage;
        statusMap[nextKey] = "in_progress";
      }
      break;
    }

    case "fail_stage": {
      const failKey = String(stage);
      retryMap[failKey] = (Number(retryMap[failKey]) || 0) + 1;
      statusMap[failKey] = "failed";
      break;
    }

    case "advance_module": {
      if (stage !== 6) {
        throw new Error(`advance_module is only valid for stage 6, got stage ${stage}`);
      }
      const k = ensureNumber(input.module_index);
      const modState = next.module_state ?? null;

      // Initial advance (k=1) seeds module_state when not yet present.
      if (!modState) {
        const total = ensureNumber(input.total_modules);
        next.module_state = {
          N: total,
          active_module: 1,
          modules_pypto_verified: [],
          module_attempts: { "1": 0 },
          module_status: { "1": "in_progress" },
        };
        if (k !== 1) {
          throw new Error(
            `advance_module: first call must set active_module=1 (got ${k}); pass total_modules=N`,
          );
        }
        break;
      }

      if (k > modState.N) {
        throw new Error(`advance_module: module_index ${k} exceeds N=${modState.N}`);
      }
      if (k <= modState.active_module && k !== modState.active_module) {
        throw new Error(
          `advance_module: cannot move backwards (active_module=${modState.active_module}, requested=${k})`,
        );
      }

      const prevActive = modState.active_module;
      const prevActiveKey = String(prevActive);
      // Mark previous module as verified and add to verified list.
      modState.module_status[prevActiveKey] = "verified";
      if (!modState.modules_pypto_verified.includes(prevActive)) {
        modState.modules_pypto_verified.push(prevActive);
      }

      if (k === prevActive) {
        // Allow idempotent advance to current (no-op for new module slot).
        break;
      }

      modState.active_module = k;
      const newKey = String(k);
      if (!(newKey in modState.module_attempts)) {
        modState.module_attempts[newKey] = 0;
      }
      modState.module_status[newKey] = "in_progress";
      // Clear last_failure on successful advance.
      next.last_failure = null;
      break;
    }

    case "record_gate": {
      const gate = ensureGate(input.gate);
      const status = ensureGateStatus(input.gate_status);
      const gateMap = next.gate_status ?? {};
      next.gate_status = gateMap;
      gateMap[String(gate)] = status;
      break;
    }

    case "record_module_attempt": {
      if (stage !== 6) {
        throw new Error(`record_module_attempt is only valid for stage 6, got stage ${stage}`);
      }
      const k = ensureNumber(input.module_index);
      const modState = next.module_state;
      if (!modState) {
        throw new Error(
          `record_module_attempt: module_state is not initialized; call advance_module(stage=6, module_index=1, total_modules=N) first`,
        );
      }
      const key = String(k);
      modState.module_attempts[key] = (Number(modState.module_attempts[key]) || 0) + 1;
      if (input.failure_category && typeof input.failure_category === "string") {
        const lf: LastFailure = {
          module: k,
          category: input.failure_category,
        };
        if (typeof input.evaluation_report_path === "string" && input.evaluation_report_path) {
          lf.evaluation_report_path = input.evaluation_report_path;
        }
        next.last_failure = lf;
      }
      // Mark blocked when cap exceeded; orchestrator decides terminal flow.
      if (modState.module_attempts[key] > PER_MODULE_ATTEMPT_CAP) {
        modState.module_status[key] = "blocked";
      } else if (!modState.module_status[key]) {
        modState.module_status[key] = "in_progress";
      }
      break;
    }

    default:
      throw new Error(`unsupported action: ${(input as { action?: string }).action ?? "unknown"}`);
  }

  // Optional subphase update may accompany any action.
  if (input.subphase !== undefined) {
    if (input.subphase !== null && !SUBPHASE_VALUES.has(input.subphase)) {
      throw new Error(`invalid subphase: ${String(input.subphase)}`);
    }
    next.current_subphase = input.subphase;
  }

  next.last_updated = new Date().toISOString();
  return next;
}

function ensureGate(val: unknown): number {
  const n = Number(val);
  if (!Number.isFinite(n) || !GATE_RANGE.has(n)) {
    throw new Error(`invalid gate: ${val} (expected 0..4)`);
  }
  return n;
}

function ensureGateStatus(val: unknown): GateStatusValue {
  if (typeof val !== "string" || !GATE_STATUS_VALUES.has(val as GateStatusValue)) {
    throw new Error(`invalid gate_status: ${val} (expected pending|in_progress|passed|failed)`);
  }
  return val as GateStatusValue;
}

/**
 * Migrate legacy state files (those written before module_state / gate_status existed).
 * Adds the new top-level keys with safe defaults so subsequent reads are well-formed.
 * Does not write to disk — caller persists if any change was made.
 */
export function migrateLegacyState(state: OrchestratorState): OrchestratorState {
  const next = JSON.parse(JSON.stringify(state)) as OrchestratorState;
  let changed = false;
  if (!next.gate_status) {
    next.gate_status = {
      "0": "pending",
      "1": "pending",
      "2": "pending",
      "3": "pending",
      "4": "pending",
    };
    changed = true;
  }
  if (next.current_subphase === undefined) {
    next.current_subphase = null;
    changed = true;
  }
  if (next.last_failure === undefined) {
    next.last_failure = null;
    changed = true;
  }
  // module_state is intentionally NOT seeded here — it is created on the first
  // advance_module call when total_modules (N) is known.
  if (changed) {
    next.last_updated = new Date().toISOString();
  }
  return next;
}
