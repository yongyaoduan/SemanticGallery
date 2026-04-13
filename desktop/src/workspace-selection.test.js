import { describe, expect, it } from "vitest";

import {
  countVisibleSelection,
  pruneSelectionToVisibleResults,
  toggleAllVisibleResults,
  toggleResultSelection
} from "./workspace-selection";

describe("workspace selection", () => {
  const results = [
    { relativePath: "cat.jpg" },
    { relativePath: "cat-friend.jpg" },
    { relativePath: "dog.jpg" }
  ];

  it("keeps bulk selection scoped to the current search result list", () => {
    const selected = new Set(["cat.jpg", "bird.jpg"]);

    const pruned = pruneSelectionToVisibleResults(results, selected);

    expect([...pruned]).toEqual(["cat.jpg"]);
    expect(countVisibleSelection(results, pruned)).toBe(1);
  });

  it("toggles all visible results without touching hidden paths", () => {
    const selected = new Set(["outside.jpg"]);

    const allVisible = toggleAllVisibleResults(results, selected);
    const cleared = toggleAllVisibleResults(results, allVisible);

    expect([...allVisible]).toEqual(["outside.jpg", "cat.jpg", "cat-friend.jpg", "dog.jpg"]);
    expect([...cleared]).toEqual(["outside.jpg"]);
  });

  it("toggles a single result path in selection mode", () => {
    const selected = new Set(["cat.jpg"]);

    expect([...toggleResultSelection(selected, "dog.jpg")]).toEqual(["cat.jpg", "dog.jpg"]);
    expect([...toggleResultSelection(selected, "cat.jpg")]).toEqual([]);
  });
});
