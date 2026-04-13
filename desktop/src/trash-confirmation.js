export function buildTrashConfirmationContent({ count = 0 } = {}) {
  if (count > 1) {
    return {
      title: `Delete ${count} images?`,
      message: "This will move the selected images to the Trash.",
      confirmLabel: "Delete"
    };
  }

  return {
    title: "Delete image?",
    message: "This will move the image to the Trash.",
    confirmLabel: "Delete"
  };
}
