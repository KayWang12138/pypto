export const SCHEMA_VERSION = 2;
export const HISTORY_CAP = 50;

export type TransitionAction = "start_stage" | "complete_stage" | "fail_stage" | "init";

export type GateSummary = {
  warnCount?: number;
  failCount?: number;
  ruleIds?: string[];
};

export type StageErrorRecord = {
  kind?: string;
  message: string;
  ruleIds?: string[];
};

export type HistoryEntry = {
  at: string;
  action: TransitionAction;
  stage: number;
  attempt: number;
  reason?: string;
  gate?: GateSummary;
  error?: StageErrorRecord;
};

export type ArtifactFingerprint = {
  sha256: string;
  stage: number;
  recorded_at: string;
};

export type TransitionInput = {
  action: TransitionAction;
  stage: number;
  reason?: string;
  gate?: GateSummary;
  error?: StageErrorRecord;
  at?: string;
};

export type OrchestratorState = {
  operator_name?: string;
  current_stage: number;
  stage_status: Record<string, string>;
  stage_retry_count?: Record<string, number>;
  last_updated?: string;

  // v2 additive fields (absent in v1 state files).
  schema_version?: number;
  history?: HistoryEntry[];
  artifacts?: Record<string, ArtifactFingerprint>;
  last_error?: StageErrorRecord & { stage: number; at: string };

  // Legacy/extra fields (e.g., spec_md_hash, perf_iteration) pass through untouched.
  [key: string]: unknown;
};

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

/**
 * Fill in v2 default fields when loading a state written by an older writer.
 * Never throws; keeps existing values intact.
 */
export function migrateState(prev: OrchestratorState): OrchestratorState {
  if (prev.schema_version === undefined) prev.schema_version = SCHEMA_VERSION;
  if (!prev.history) prev.history = [];
  if (!prev.artifacts) prev.artifacts = {};
  return prev;
}

function pushHistory(state: OrchestratorState, entry: HistoryEntry): void {
  const list = state.history ?? (state.history = []);
  list.push(entry);
  while (list.length > HISTORY_CAP) list.shift();
}

export function applyTransition(
  prev: OrchestratorState,
  input: TransitionInput,
): OrchestratorState {
  const stage = ensureNumber(input.stage);
  const next = cloneState(prev);
  migrateState(next);
  const statusMap = next.stage_status ?? {};
  next.stage_status = statusMap;
  const retryMap = next.stage_retry_count ?? {};
  next.stage_retry_count = retryMap;
  const now = input.at ?? new Date().toISOString();
  // Attempt counter snapshot BEFORE any retry_count increment in fail_stage.
  // Rule: attempt = retries_seen_so_far + 1 (first start is attempt 1).
  const attempt = (Number(retryMap[String(stage)]) || 0) + 1;

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
      // Clear last_error if it belongs to the stage we just completed.
      if (next.last_error && next.last_error.stage === stage) {
        delete next.last_error;
      }
      break;
    }

    case "fail_stage": {
      const failKey = String(stage);
      retryMap[failKey] = (Number(retryMap[failKey]) || 0) + 1;
      statusMap[failKey] = "failed";
      next.last_error = {
        stage,
        at: now,
        kind: input.error?.kind,
        message: input.error?.message ?? input.reason ?? "stage failed",
        ...(input.error?.ruleIds ? { ruleIds: input.error.ruleIds } : {}),
      };
      break;
    }

    default:
      throw new Error(`unsupported action: ${(input as { action?: string }).action ?? "unknown"}`);
  }

  // Record transition in rolling history (bounded by HISTORY_CAP).
  pushHistory(next, {
    at: now,
    action: input.action,
    stage,
    attempt,
    reason: input.reason,
    gate: input.gate,
    error: input.action === "fail_stage" ? next.last_error : undefined,
  });

  next.schema_version = SCHEMA_VERSION;
  next.last_updated = now;
  return next;
}
