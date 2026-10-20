// @ts-check
const { existsSync, readFileSync, statSync } = require("node:fs");
const path = require("node:path");

const {
  isLikelyPackagedToolsDir,
  runtimeOverrideEnv,
} = require("../simple-page-child-env.cjs");
const { createDetailedError } = require("../simple-page-runtime-common.cjs");
const { resolveToolsDir } = require("../model/runtime-locations.cjs");

/** @typedef {import("../runtime-jsdoc-types").RuntimeOptions & { ffmpegPath?: string | null; toolsDir?: string | null }} RuntimePathOptions */

/** @param {string} toolsDir @returns {string | null} */
function systemFfmpegCandidate(toolsDir) {
  try {
    const metadata = JSON.parse(
      readFileSync(
        path.join(toolsDir || "", "ffmpeg", "ffmpeg-runtime.json"),
        "utf8",
      ),
    );
    return metadata?.source === "system" &&
      typeof metadata.path === "string"
      ? metadata.path
      : null;
  } catch {
    return null;
  }
}

/** @param {string} toolsDir */
function bundledFfmpegCandidates(toolsDir) {
  const binaryName = process.platform === "win32" ? "ffmpeg.exe" : "ffmpeg";
  return [
    systemFfmpegCandidate(toolsDir),
    path.join(toolsDir || "", "ffmpeg", binaryName),
    path.join(toolsDir || "", "ffmpeg", "bin", binaryName),
    path.join(toolsDir || "", binaryName),
  ].filter((candidate) => typeof candidate === "string");
}

/** @param {RuntimePathOptions} [options] */
function resolveFfmpegPath(options = {}) {
  const toolsDir = resolveToolsDir(options);
  const candidates = bundledFfmpegCandidates(toolsDir);
  const bundledPath = candidates.find(isExistingFilePath);
  if (bundledPath) return bundledPath;
  if (isLikelyPackagedToolsDir(toolsDir)) {
    throw createDetailedError(
      "Bundled ffmpeg is missing from the packaged tools directory.",
      {
        toolsDir,
        candidatePaths: candidates,
        command: "ffmpeg",
      },
    );
  }
  const explicit = [
    options.ffmpegPath,
    runtimeOverrideEnv("MANGA_TRANSLATOR_FFMPEG_PATH", options),
  ].find(isExistingFilePath);
  return (
    explicit ||
    resolveDevelopmentFfmpegPath() ||
    (process.platform === "win32" ? "ffmpeg.exe" : "ffmpeg")
  );
}

/** @returns {string | null} */
function resolveDevelopmentFfmpegPath() {
  return (
    bundledFfmpegCandidates(path.join(process.cwd(), "tools")).find(
      isExistingFilePath,
    ) ?? null
  );
}

/** @param {unknown} candidate */
function isExistingFilePath(candidate) {
  return (
    typeof candidate === "string" &&
    existsSync(candidate) &&
    statSync(candidate).isFile()
  );
}

module.exports = { resolveDevelopmentFfmpegPath, resolveFfmpegPath };
