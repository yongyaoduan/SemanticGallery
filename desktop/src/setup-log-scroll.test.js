import { describe, expect, it } from "vitest";

import { nextSetupLogPinState, shouldAutoScrollSetupLogs } from "./setup-log-scroll";

describe("setup log scroll policy", () => {
  it("keeps auto-scroll enabled while the log view stays near the bottom", () => {
    expect(
      nextSetupLogPinState({
        scrollTop: 240,
        clientHeight: 120,
        scrollHeight: 368
      })
    ).toBe(true);
  });

  it("stops auto-scroll after the user scrolls away from the bottom", () => {
    expect(
      nextSetupLogPinState({
        scrollTop: 40,
        clientHeight: 120,
        scrollHeight: 368
      })
    ).toBe(false);
  });

  it("only forces the latest log line into view when the list is pinned", () => {
    expect(
      shouldAutoScrollSetupLogs({
        pinnedToBottom: true,
        scrollTop: 40,
        clientHeight: 120,
        scrollHeight: 368
      })
    ).toBe(true);

    expect(
      shouldAutoScrollSetupLogs({
        pinnedToBottom: false,
        scrollTop: 40,
        clientHeight: 120,
        scrollHeight: 368
      })
    ).toBe(false);
  });
});
