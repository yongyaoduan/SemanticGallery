import { describe, expect, it } from "vitest";

import { createInitialState, reduceAction } from "./state";

describe("desktop state", () => {
  it("shows setup progress and refresh state independently", () => {
    let state = createInitialState();
    state = reduceAction(state, { type: "setup-progress", payload: { currentStep: 2, totalSteps: 5 } });
    state = reduceAction(state, { type: "refresh-started" });

    expect(state.setup.currentStep).toBe(2);
    expect(state.setup.totalSteps).toBe(5);
    expect(state.refresh.isRunning).toBe(true);
  });
});
