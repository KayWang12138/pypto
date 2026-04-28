import { describe, test, expect } from "bun:test";
import {
  applyTransition,
  migrateLegacyState,
  type OrchestratorState,
} from "../lib/state-transition-core";

function makeState(overrides: Partial<OrchestratorState> = {}): OrchestratorState {
  return {
    operator_name: "test_op",
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

  test("rejects wrong stage", () => {
    expect(() =>
      applyTransition(
        { current_stage: 2, stage_status: { "2": "in_progress" } } as OrchestratorState,
        { action: "complete_stage", stage: 1 },
      ),
    ).toThrow("current_stage");
  });

  test("rejects non-in_progress stage", () => {
    expect(() =>
      applyTransition(
        { current_stage: 2, stage_status: { "2": "pending" } } as OrchestratorState,
        { action: "complete_stage", stage: 2 },
      ),
    ).toThrow("in_progress");
  });

  test("does not advance past last stage", () => {
    const state: OrchestratorState = {
      current_stage: 3,
      stage_status: { "1": "completed", "2": "completed", "3": "in_progress" },
    };
    const next = applyTransition(state, { action: "complete_stage", stage: 3 });
    expect(next.stage_status["3"]).toBe("completed");
    expect(next.current_stage).toBe(3);
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
    expect(next.current_stage).toBe(2);
  });

  test("rejects if previous stage not completed", () => {
    const state: OrchestratorState = {
      operator_name: "test",
      current_stage: 2,
      stage_status: { "1": "pending", "2": "pending" },
    };
    expect(() =>
      applyTransition(state, { action: "start_stage", stage: 2 }),
    ).toThrow();
  });

  test("rejects starting a completed stage", () => {
    const state = makeState({
      current_stage: 2,
      stage_status: { "1": "completed", "2": "completed", "3": "pending" },
    });
    expect(() =>
      applyTransition(state, { action: "start_stage", stage: 2 }),
    ).toThrow("already completed");
  });
});

describe("fail_stage", () => {
  test("marks stage as failed and increments retry", () => {
    const result = applyTransition(makeState(), { action: "fail_stage", stage: 2 });
    expect(result.stage_status["2"]).toBe("failed");
  });
});

describe("advance_module", () => {
  function stage6State(overrides: Partial<OrchestratorState> = {}): OrchestratorState {
    return {
      operator_name: "test_op",
      current_stage: 6,
      stage_status: {
        "1": "completed",
        "2": "completed",
        "3": "completed",
        "4": "completed",
        "5": "completed",
        "6": "in_progress",
        "7": "pending",
      },
      stage_retry_count: {},
      ...overrides,
    };
  }

  test("first call seeds module_state with N and active_module=1", () => {
    const state = stage6State();
    const next = applyTransition(state, {
      action: "advance_module",
      stage: 6,
      module_index: 1,
      total_modules: 3,
    });
    expect(next.module_state).toBeDefined();
    expect(next.module_state!.N).toBe(3);
    expect(next.module_state!.active_module).toBe(1);
    expect(next.module_state!.modules_pypto_verified).toEqual([]);
    expect(next.module_state!.module_status["1"]).toBe("in_progress");
    expect(next.module_state!.module_attempts["1"]).toBe(0);
  });

  test("first call rejects module_index != 1", () => {
    expect(() =>
      applyTransition(stage6State(), {
        action: "advance_module",
        stage: 6,
        module_index: 2,
        total_modules: 3,
      }),
    ).toThrow("first call must set active_module=1");
  });

  test("first call rejects missing total_modules", () => {
    expect(() =>
      applyTransition(stage6State(), {
        action: "advance_module",
        stage: 6,
        module_index: 1,
      }),
    ).toThrow();
  });

  test("subsequent advance moves active_module forward", () => {
    const seeded = applyTransition(stage6State(), {
      action: "advance_module",
      stage: 6,
      module_index: 1,
      total_modules: 3,
    });
    const next = applyTransition(seeded, {
      action: "advance_module",
      stage: 6,
      module_index: 2,
    });
    expect(next.module_state!.active_module).toBe(2);
    expect(next.module_state!.modules_pypto_verified).toEqual([1]);
    expect(next.module_state!.module_status["1"]).toBe("verified");
    expect(next.module_state!.module_status["2"]).toBe("in_progress");
    expect(next.module_state!.module_attempts["2"]).toBe(0);
  });

  test("clears last_failure on successful advance", () => {
    const seeded = applyTransition(stage6State(), {
      action: "advance_module",
      stage: 6,
      module_index: 1,
      total_modules: 2,
    });
    const withFailure: OrchestratorState = {
      ...seeded,
      last_failure: { module: 1, category: "precision" },
    };
    const next = applyTransition(withFailure, {
      action: "advance_module",
      stage: 6,
      module_index: 2,
    });
    expect(next.last_failure).toBeNull();
  });

  test("rejects module_index > N", () => {
    const seeded = applyTransition(stage6State(), {
      action: "advance_module",
      stage: 6,
      module_index: 1,
      total_modules: 2,
    });
    expect(() =>
      applyTransition(seeded, {
        action: "advance_module",
        stage: 6,
        module_index: 3,
      }),
    ).toThrow("exceeds N=2");
  });

  test("rejects backwards advance", () => {
    let s = applyTransition(stage6State(), {
      action: "advance_module",
      stage: 6,
      module_index: 1,
      total_modules: 3,
    });
    s = applyTransition(s, {
      action: "advance_module",
      stage: 6,
      module_index: 2,
    });
    expect(() =>
      applyTransition(s, {
        action: "advance_module",
        stage: 6,
        module_index: 1,
      }),
    ).toThrow("cannot move backwards");
  });

  test("rejects when stage != 6", () => {
    expect(() =>
      applyTransition(stage6State({ current_stage: 5 }), {
        action: "advance_module",
        stage: 5,
        module_index: 1,
        total_modules: 1,
      }),
    ).toThrow("only valid for stage 6");
  });
});

describe("record_gate", () => {
  test("sets gate status", () => {
    const state = makeState();
    const next = applyTransition(state, {
      action: "record_gate",
      stage: 1,
      gate: 2,
      gate_status: "passed",
    });
    expect(next.gate_status?.["2"]).toBe("passed");
  });

  test("rejects invalid gate number", () => {
    expect(() =>
      applyTransition(makeState(), {
        action: "record_gate",
        stage: 1,
        gate: 5,
        gate_status: "passed",
      }),
    ).toThrow("invalid gate");
  });

  test("rejects invalid gate_status", () => {
    expect(() =>
      applyTransition(makeState(), {
        action: "record_gate",
        stage: 1,
        gate: 0,
        // @ts-expect-error invalid value on purpose
        gate_status: "weird",
      }),
    ).toThrow("invalid gate_status");
  });
});

describe("record_module_attempt", () => {
  function seededState(): OrchestratorState {
    const base: OrchestratorState = {
      operator_name: "test_op",
      current_stage: 6,
      stage_status: {
        "1": "completed",
        "2": "completed",
        "3": "completed",
        "4": "completed",
        "5": "completed",
        "6": "in_progress",
        "7": "pending",
      },
    };
    return applyTransition(base, {
      action: "advance_module",
      stage: 6,
      module_index: 1,
      total_modules: 2,
    });
  }

  test("increments module_attempts and stores last_failure", () => {
    const next = applyTransition(seededState(), {
      action: "record_module_attempt",
      stage: 6,
      module_index: 1,
      failure_category: "precision",
      evaluation_report_path: "custom/test_op/eval/evaluation_report.json",
    });
    expect(next.module_state!.module_attempts["1"]).toBe(1);
    expect(next.last_failure).toEqual({
      module: 1,
      category: "precision",
      evaluation_report_path: "custom/test_op/eval/evaluation_report.json",
    });
  });

  test("marks module blocked when cap exceeded", () => {
    let s = seededState();
    for (let i = 0; i < 4; i++) {
      s = applyTransition(s, {
        action: "record_module_attempt",
        stage: 6,
        module_index: 1,
        failure_category: "precision",
      });
    }
    expect(s.module_state!.module_attempts["1"]).toBe(4);
    expect(s.module_state!.module_status["1"]).toBe("blocked");
  });

  test("rejects when module_state not initialized", () => {
    const base: OrchestratorState = {
      current_stage: 6,
      stage_status: { "6": "in_progress" },
    };
    expect(() =>
      applyTransition(base, {
        action: "record_module_attempt",
        stage: 6,
        module_index: 1,
      }),
    ).toThrow("module_state is not initialized");
  });

  test("rejects when stage != 6", () => {
    const seeded = seededState();
    expect(() =>
      applyTransition(seeded, {
        action: "record_module_attempt",
        stage: 5,
        module_index: 1,
      }),
    ).toThrow("only valid for stage 6");
  });
});

describe("subphase update (auxiliary)", () => {
  test("can be set on any action", () => {
    const next = applyTransition(makeState({ current_stage: 6, stage_status: { "5": "completed", "6": "in_progress" } }), {
      action: "record_gate",
      stage: 6,
      gate: 3,
      gate_status: "in_progress",
      subphase: "verifying",
    });
    expect(next.current_subphase).toBe("verifying");
  });

  test("can be cleared with null", () => {
    const seed = applyTransition(makeState({ current_stage: 6, stage_status: { "5": "completed", "6": "in_progress" } }), {
      action: "record_gate",
      stage: 6,
      gate: 3,
      gate_status: "passed",
      subphase: "verifying",
    });
    const next = applyTransition(seed, {
      action: "record_gate",
      stage: 6,
      gate: 4,
      gate_status: "in_progress",
      subphase: null,
    });
    expect(next.current_subphase).toBeNull();
  });

  test("rejects invalid subphase value", () => {
    expect(() =>
      applyTransition(makeState(), {
        action: "record_gate",
        stage: 1,
        gate: 0,
        gate_status: "passed",
        // @ts-expect-error
        subphase: "bogus",
      }),
    ).toThrow("invalid subphase");
  });
});

describe("migrateLegacyState", () => {
  test("seeds gate_status, current_subphase, last_failure when missing", () => {
    const legacy: OrchestratorState = {
      operator_name: "old_op",
      current_stage: 3,
      stage_status: { "1": "completed", "2": "completed", "3": "in_progress" },
    };
    const migrated = migrateLegacyState(legacy);
    expect(migrated.gate_status).toEqual({
      "0": "pending",
      "1": "pending",
      "2": "pending",
      "3": "pending",
      "4": "pending",
    });
    expect(migrated.current_subphase).toBeNull();
    expect(migrated.last_failure).toBeNull();
  });

  test("preserves existing values", () => {
    const state: OrchestratorState = {
      current_stage: 6,
      stage_status: {},
      gate_status: { "0": "passed", "1": "passed", "2": "in_progress", "3": "pending", "4": "pending" },
      current_subphase: "verifying",
      last_failure: { module: 2, category: "precision" },
    };
    const migrated = migrateLegacyState(state);
    expect(migrated.gate_status?.["2"]).toBe("in_progress");
    expect(migrated.current_subphase).toBe("verifying");
    expect(migrated.last_failure).toEqual({ module: 2, category: "precision" });
  });

  test("does not seed module_state (created on first advance_module)", () => {
    const legacy: OrchestratorState = {
      current_stage: 6,
      stage_status: { "6": "in_progress" },
    };
    const migrated = migrateLegacyState(legacy);
    expect(migrated.module_state).toBeUndefined();
  });
});

describe("Stage 5 precision pass → complete 5 then complete 6", () => {
  test("orchestrator can complete_stage(5) then complete_stage(6) to reach Stage 7", () => {
    const initial: OrchestratorState = {
      operator_name: "test_op",
      current_stage: 5,
      stage_status: {
        "1": "completed",
        "2": "completed",
        "3": "completed",
        "4": "completed",
        "5": "in_progress",
        "6": "pending",
        "7": "pending",
      },
    };

    // Step 1: complete stage 5 (precision passed)
    const after5 = applyTransition(initial, { action: "complete_stage", stage: 5 });
    expect(after5.stage_status["5"]).toBe("completed");
    expect(after5.stage_status["6"]).toBe("in_progress");
    expect(after5.current_stage).toBe(6);

    // Step 2: immediately complete stage 6 (no precision fix needed)
    const after6 = applyTransition(after5, { action: "complete_stage", stage: 6 });
    expect(after6.stage_status["6"]).toBe("completed");
    expect(after6.stage_status["7"]).toBe("in_progress");
    expect(after6.current_stage).toBe(7);
  });
});
