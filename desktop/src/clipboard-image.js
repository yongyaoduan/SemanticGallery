export function extractImageFileFromClipboardItems(items) {
  for (const item of items ?? []) {
    if (item?.kind === "file" && typeof item.type === "string" && item.type.startsWith("image/")) {
      const file = item.getAsFile?.();
      if (file) {
        return file;
      }
    }
  }
  return null;
}
