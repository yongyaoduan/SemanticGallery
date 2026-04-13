import { describe, expect, it } from "vitest";

import { buildIndexProgressTiming, buildIndexProgressSummary } from "./index-progress";

describe("index progress helpers", () => {
  it("formats elapsed and remaining time for an active indexing run", () => {
    expect(
      buildIndexProgressTiming({
        status: "running",
        elapsedSeconds: 42,
        remainingSeconds: 93
      })
    ).toEqual({
      elapsedLabel: "Elapsed 00:42",
      remainingLabel: "Left 01:33"
    });
  });

  it("keeps completed indexing runs at full width and clears the remaining time", () => {
    expect(
      buildIndexProgressSummary({
        status: "ready",
        current: 7062,
        total: 7062,
        elapsedSeconds: 315,
        remainingSeconds: 0
      })
    ).toEqual({
      countLabel: "7062 / 7062",
      percent: 100,
      elapsedLabel: "Elapsed 05:15",
      remainingLabel: ""
    });
  });
});
