import { createHash, randomUUID } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, it, vi } from "vitest";
import manifest from "../src/main/pipeline/fontChapterC18Manifest.json";
import {
  prepareFontChapterRuntime,
  resolveFontChapterRuntimeManifest,
} from "../src/main/pipeline/fontChapterRuntimeAssets";

const download = vi.hoisted(() =>
  vi.fn().mockRejectedValue(new Error("network unavailable")),
);
vi.mock("../src/main/runtimeSupport/modelDownloads", () => ({
  ensureRemoteFile: download,
}));

describe("approved C23 runtime binding", () => {
  it("selects the matching pinned pack for existing and fresh Windows installations", () => {
    const oldPython = resolveFontChapterRuntimeManifest(
      "win32",
      "x64",
      "cpython-312",
    );
    const newPython = resolveFontChapterRuntimeManifest(
      "win32",
      "x64",
      "cpython-314",
    );
    expect(oldPython.url).toContain("/font-chapter-c23-20260909-r2/");
    expect(oldPython.archive.sha256).toBe(
      "302b6ab214ea4cadac864eaeb42cbb30c44ca30377434b2327fd0fa7fd9be360",
    );
    expect(oldPython.manifest.files.at(-1)?.sha256).toBe(
      "6f1b97543c45dbb7d8ff411341162fb3a0b07b49b6c071d3e6e80f2ba1a41e3e",
    );
    expect(newPython.url).toContain("/font-chapter-c23-20261008-r3/");
    expect(oldPython.manifest.assetDirectory).not.toBe(
      newPython.manifest.assetDirectory,
    );
    expect(oldPython.manifest.files.slice(0, -1)).toEqual(
      newPython.manifest.files.slice(0, -1),
    );
    expect(oldPython.manifest.runtimeSources).toEqual(
      newPython.manifest.runtimeSources,
    );
  });
  it.each(["cpython-311", "cpython-313", "cpython-314t", "pypy-312", ""])(
    "rejects unsupported ABI %s before downloading or running inference",
    async (pythonAbi) => {
      download.mockClear();
      expect(() =>
        resolveFontChapterRuntimeManifest("win32", "x64", pythonAbi),
      ).toThrow("Unsupported font matching Python ABI");
      await expect(
        prepareFontChapterRuntime({
          paths: { dataRoot: "unused", runtimeDir: "unused" },
          platform: "win32",
          arch: "x64",
          pythonAbi,
        }),
      ).rejects.toThrow("Unsupported font matching Python ABI");
      expect(download).not.toHaveBeenCalled();
    },
  );
  it("passes the actual release digest, cancellation and progress to the installer", async () => {
    const options = {
      platform: "win32" as const,
      arch: "x64",
      pythonAbi: "cpython-314",
      paths: {
        dataRoot: resolve(tmpdir(), `fc23-${randomUUID()}`),
        runtimeDir: resolve("src/main/runtime"),
      },
      signal: new AbortController().signal,
      onProgress: vi.fn(),
    };
    await expect(prepareFontChapterRuntime(options)).rejects.toThrow(
      "network unavailable",
    );
    const resolved = resolveFontChapterRuntimeManifest(
      "win32",
      "x64",
      "cpython-314",
    );
    expect(download).toHaveBeenCalledWith(
      expect.objectContaining({
        url: resolved.url,
        expectedSha256: resolved.archive.sha256,
        expectedTotalBytes: resolved.archive.bytes,
        signal: options.signal,
        onProgress: options.onProgress,
      }),
    );
  });
  it("uses separate native dependencies while preserving the same trained models and algorithms", () => {
    const windows = resolveFontChapterRuntimeManifest("win32", "x64");
    const mac = resolveFontChapterRuntimeManifest("darwin", "arm64");
    expect(windows.manifest.runtimeSources).toEqual(manifest.runtimeSources);
    expect(mac.manifest.runtimeSources).toEqual(manifest.runtimeSources);
    const trained = manifest.files.filter(
      (row) => row.path !== "python-inventory.json",
    );
    for (const resolved of [windows, mac]) {
      expect(resolved.manifest.files.slice(0, trained.length)).toEqual(trained);
      expect(resolved.url).toContain("/font-chapter-c23-20261008-r3/");
      expect(resolved.archive.bytes).toBeGreaterThan(1_000_000);
      expect(resolved.manifest.assetDirectory).not.toBe(
        manifest.assetDirectory,
      );
    }
    expect(windows.manifest.files.at(-1)?.sha256).not.toBe(
      mac.manifest.files.at(-1)?.sha256,
    );
    expect(() => resolveFontChapterRuntimeManifest("linux", "x64")).toThrow(
      "Unsupported",
    );
  });
  it("binds every shipped algorithm and keeps a distinct rollback-safe asset root", () => {
    expect(manifest.version).toBe("c23.0");
    expect(manifest.assetDirectory).toBe("font-chapter-c18/c23-v1");
    const root = resolve("src/main/runtime/font-chapter-c18");
    expect(manifest.runtimeSources.map((row) => row.path).sort()).toEqual(
      readdirSync(root)
        .filter((name) => name.endsWith(".py"))
        .sort(),
    );
    for (const row of manifest.runtimeSources) {
      const bytes = readFileSync(resolve(root, row.path));
      expect(createHash("sha256").update(bytes).digest("hex")).toBe(row.sha256);
    }
  });
});
