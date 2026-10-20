import type { AppPaths } from "../appPaths";
import type { RuntimeAssetProgress } from "../runtimeSupport/modelDownloads";
import approved from "./fontChapterC18Manifest.json";
import release from "./fontChapterReleaseManifest.json";
import { installFontChapterAssets } from "./fontChapterAssetInstaller";

type SupportedFontChapterRuntimeKey = "win32-x64" | "darwin-arm64";

function isSupportedFontChapterRuntimeKey(
  key: string,
): key is SupportedFontChapterRuntimeKey {
  return key === "win32-x64" || key === "darwin-arm64";
}

export function isFontChapterRuntimeSupported(
  platform: NodeJS.Platform = process.platform,
  arch: string = process.arch,
): boolean {
  return isSupportedFontChapterRuntimeKey(`${platform}-${arch}`);
}

export function resolveFontChapterRuntimeManifest(
  platform: NodeJS.Platform = process.platform,
  arch: string = process.arch,
  pythonAbi: string = platform === "win32" ? "cpython-314" : "cpython-312",
) {
  const key = `${platform}-${arch}`;
  if (!isSupportedFontChapterRuntimeKey(key))
    throw new Error(`Unsupported font matching platform: ${key}`);
  if (
    pythonAbi !== "cpython-312" &&
    !(key === "win32-x64" && pythonAbi === "cpython-314")
  )
    throw new Error(
      `Unsupported font matching Python ABI: ${key}/${pythonAbi}`,
    );
  const selected =
    key === "win32-x64" && pythonAbi === "cpython-312"
      ? release.windowsPython312
      : {
          tag: release.tag,
          assetRoot: release.assetRoot,
          ...release.platforms[key],
        };
  const entry = selected;
  const manifest = {
    ...approved,
    assetDirectory: `${selected.assetRoot}/${key}`,
    files: [
      ...approved.files.filter((row) => row.path !== "python-inventory.json"),
      ...release.licenseFiles,
      entry.pythonInventory,
    ],
  };
  const url = `https://github.com/ucx0204/CarrotMangaTranslator/releases/download/${selected.tag}/${entry.archive.path}`;
  return { manifest, archive: entry.archive, url };
}

export async function prepareFontChapterRuntime(options: {
  paths: Pick<AppPaths, "dataRoot" | "runtimeDir">;
  pythonAbi: string;
  platform?: NodeJS.Platform;
  arch?: string;
  signal?: AbortSignal;
  onProgress?: (progress: RuntimeAssetProgress) => void;
}) {
  const resolved = resolveFontChapterRuntimeManifest(
    options.platform ?? process.platform,
    options.arch ?? process.arch,
    options.pythonAbi,
  );
  const assets = await installFontChapterAssets({ ...options, ...resolved });
  return { assets, manifest: resolved.manifest };
}
