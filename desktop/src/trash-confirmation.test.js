import { describe, expect, it } from "vitest";

import { buildTrashConfirmationContent } from "./trash-confirmation";

describe("trash confirmation", () => {
  it("builds single-image confirmation copy", () => {
    expect(buildTrashConfirmationContent({ fileName: "cat.jpg" })).toEqual({
      title: "Delete image?",
      message: "This will move the image to the Trash.",
      confirmLabel: "Delete"
    });
  });

  it("builds multi-image confirmation copy", () => {
    expect(buildTrashConfirmationContent({ count: 3 })).toEqual({
      title: "Delete 3 images?",
      message: "This will move the selected images to the Trash.",
      confirmLabel: "Delete"
    });
  });
});
