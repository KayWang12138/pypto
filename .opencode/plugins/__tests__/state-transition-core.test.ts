import { describe, test, expect } from "bun:test";
import {
  applyTransition,
  migrateState,
  SCHEMA_VERSION,
  HISTORY_CAP,
  type OrchestratorState,
} from "../lib/state-transition-core";

function makeState(overrides: Partial<OrchestratorState> = {}): OrchestratorState {
  return {
    operator_name: "x",
    current_stage: 1,
    stage_status: {
      "1": "completed",
      "2": "pending",
      "3": "pending",
      "4": "pending",
      "5": "pending",
      "6": "pending",
      "7": "pending",
    },
    stage_retry_count: {},
    ...overrides,
  };
}

describe("complete_stage", () => {
  test("advances to next stage", () => {
    const state = makeState({
      current_stage: 2,
      stage_status: { "1": "completed", "2": "in_progress", "3": "pending" },
    });
    const next = applyTransition(state, { action: "complete_stage", stage: 2 });
    expect(next.stage_status["2"]).toBe("completed");
    expect(next.stage_status["3"]).toBe("in_progress");
    expect(next.current_stage).toBe(3);
  });

  test("throws when stage doesn't match current_stage", () => {
    expect(() =>
      applyTransition(
        { current_stage: 2, stage_status: { "2": "in_progress" } } as OrchestratorState,
        { action: "complete_stage", stage: 1 },
      ),
    ).toThrow("current_stage");
  });

  test("throws when target stage is not in_progress", () => {
    expect(() =>
      applyTransition(
        { current_stage: 2, stage_status: { "2": "pending" } } as OrchestratorState,
        { action: "complete_stage", stage: 2 },
      ),
    ).toThrow("not \"in_progress\"");
  });

  test("stops advancing past last stage", () => {
    const state: OrchestratorState = {
      current_stage: 7,
      stage_status: { "1": "completed", "2": "completed", "3": "completed", "4": "completed", "5": "completed", "6": "completed", "7": "in_progress" },
    };
    const next = applyTransition(state, { action: "complete_stage", stage: 7 });
    expect(next.stage_status["7"]).toBe("completed");
    expect(next.current_stage).toBe(7);
  });
});

describe("start_stage", () => {
  test("starts a failed stage for retry", () => {
    const state: OrchestratorState = {
      current_stage: 2,
      stage_status: { "1": "completed", "2": "failed" },
      stage_retry_count: { "2": 1 },
    };
    const next = applyTransition(state, { action: "start_stage", stage: 2 });
    expect(next.stage_status["2"]).toBe("in_progress");
  });

  test("rejects when previous stage incomplete", () => {
    const state: OrchestratorState = {
      current_stage: 1,
      stage_status: { "1": "pending" },
    };
    expect(() => applyTransition(state, { action: "start_stage", stage: 2 })).toThrow(
      "previous stage",
    );
  });

  test("rejects already completed stage", () => {
    const state: OrchestratorState = {
      current_stage: 3,
      stage_status: { "1": "completed", "2": "completed", "3": "completed" },
    };
    expect(() => applyTransition(state, { action: "start_stage", stage: 3 })).toThrow(
      "already completed",
    );
  });
});

describe("fail_stage", () => {
  test("marks stage as failed and bumps retry counter", () => {
    const state = makeState({ current_stage: 2, stage_status: { "1": "completed", "2": "in_progress" } });
    const result = applyTransition(state, { action: "fail_stage", stage: 2, reason: "precision fails" });
    expect(result.stage_status["2"]).toBe("failed");
    expect(result.stage_retry_count?.["2"]).toBe(1);
    expect(result.last_error?.stage).toBe(2);
    expect(result.last_error?.message).toBe("precision fails");
  });
});

describe("v2 additions: schema_version + history + artifacts + last_error", () => {
  test("applyTransition stamps schema_version and seeds history", () => {
    const initial: OrchestratorState = {
      current_stage: 0,
      stage_status: {},
    };
    const next = applyTransition(initial, { action: "init", stage: 1 });
    expect(next.schema_version).toBe(2);
    expect(next.history).toBeDefined();
    expect(next.history!.length).toBe(1);
    expect(next.history![0].action).toBe("init");
    expect(next.history![0].stage).toBe(1);
    expect(next.history![0].attempt).toBe(1);
  });

  test("fail_stage records last_error and captures it in history", () => {
    const state = makeState({ current_stage: 3, stage_status: { "3": "in_progress" } });
    const next = applyTransition(state, {
      action: "fail_stage",
      stage: 3,
      reason: "gate OL01 failed",
      error: { kind: "lint-violation", message: "D1 rule failed", ruleIds: ["OL01"] },
    });
    expect(next.last_error).toBeDefined();
    expect(next.last_error!.stage).toBe(3);
    expect(next.last_error!.message).toBe("D1 rule failed");
    expect(next.history![0].error?.ruleIds).toEqual(["OL01"]);
  });

  test("complete_stage after fail clears last_error for that stage", () => {
    const state = makeState({
      current_stage: 3,
      stage_status: { "1": "completed", "2": "completed", "3": "in_progress" },
    });
    const afterFail = applyTransition(state, {
      action: "fail_stage",
      stage: 3,
      reason: "boom",
    });
    // retry by starting again
    const restarted = applyTransition(afterFail, { action: "start_stage", stage: 3 });
    // then succeed
    const afterComplete = applyTransition(restarted, { action: "complete_stage", stage: 3 });
    expect(afterComplete.last_error).toBeUndefined();
  });

  test("history is capped at HISTORY_CAP entries", () => {
    let state = makeState({
      current_stage: 3,
      stage_status: { "1": "completed", "2": "completed", "3": "in_progress" },
    });
    // Fail and re-start to generate many history entries
    for (let i = 0; i < HISTORY_CAP + 10; i++) {
      state = applyTransition(state, { action: "fail_stage", stage: 3 });
      state = applyTransition(state, { action: "start_stage", stage: 3 });
    }
    expect(state.history!.length).toBe(HISTORY_CAP);
  });

  test("migrateState preserves existing fields and fills defaults", () => {
    const legacy: OrchestratorState = {
      operator_name: "foo",
      current_stage: 2,
      stage_status: { "1": "completed", "2": "in_progress" },
    };
    const migrated = migrateState({ ...legacy });
    expect(migrated.schema_version).toBe(SCHEMA_VERSION);
    expect(migrated.history).toEqual([]);
    expect(migrated.artifacts).toEqual({});
    expect(migrated.operator_name).toBe("foo");
    expect(migrated.current_stage).toBe(legacy.current_stage);
  });
});
