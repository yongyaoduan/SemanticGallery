import { describe, expect, it } from "vitest";

import { createInitialState, reduceAction } from "./state";

describe("desktop state", () => {
  it("tracks setup steps, logs, and refresh state independently", () => {
    let state = createInitialState();
    state = reduceAction(state, {
      type: "setup-progress",
      payload: { task: "prepare-base-model", phase: "start", current: 2, total: 5 }
    });
    state = reduceAction(state, { type: "setup-log", payload: { line: "Downloading weights" } });
    state = reduceAction(state, { type: "refresh-started" });

    expect(state.setup.currentStep).toBe(2);
    expect(state.setup.totalSteps).toBe(5);
    expect(state.setup.steps.find((step) => step.task === "prepare-base-model")?.status).toBe("running");
    expect(state.setup.logs).toEqual(["Downloading weights"]);
    expect(state.refresh.isRunning).toBe(true);
  });
});
