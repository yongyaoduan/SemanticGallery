import { describe, expect, it } from "vitest";
import config from "../vite.config.js";

describe("desktop build config", () => {
  it("uses relative asset paths for the packaged app", () => {
    expect(config.base).toBe("./");
  });
});
