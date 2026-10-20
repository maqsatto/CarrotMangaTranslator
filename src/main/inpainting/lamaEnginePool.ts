import type { AppPaths } from "../appPaths";
import { releaseModelResource } from "../runtimeSupport/modelCleanupBarrier";
import { LeasedIdleResourcePool } from "../runtimeSupport/leasedIdleResource";
import { normalizeComputeGpuIndex } from "../../shared/gpuSettings";
import type {
  InpaintingModel,
  KoharuInpaintingBackend,
} from "../../shared/inpaintingSettingsTypes";
import type {
  InpaintingEngine,
  InpaintingRuntimeProgress,
} from "./inpaintingEngine";
import {
  logInpaintingRuntimeError,
  logInpaintingRuntimeInfo,
} from "./inpaintingRuntimeLogger";
import {
  prepareLamaInpaintingEngine,
  resolveLamaEnginePaths,
  type LamaInpaintingEngine,
} from "./lamaEngine";

const LAMA_ENGINE_IDLE_TTL_MS = 30 * 1000;

export type LamaEngineLease = {
  engine: InpaintingEngine;
  release: () => void;
};

type AcquireLamaEngineOptions = {
  appPaths: AppPaths;
  model: Extract<InpaintingModel, "lama-manga">;
  backend: KoharuInpaintingBackend;
  computeGpuIndex?: number;
  signal?: AbortSignal;
  onProgress?: (progress: InpaintingRuntimeProgress) => void;
};

const lamaEnginePool = new LeasedIdleResourcePool<LamaInpaintingEngine>({
  idleTtlMs: LAMA_ENGINE_IDLE_TTL_MS,
  isReusable: (engine) => engine.isHealthy?.() !== false,
  dispose: disposeLamaEngine,
});

export async function acquireLamaInpaintingEngine(
  options: AcquireLamaEngineOptions,
): Promise<LamaEngineLease> {
  const computeGpuIndex = normalizeComputeGpuIndex(options.computeGpuIndex);
  const paths = resolveLamaEnginePaths(options.appPaths, options.backend);
  const key = [
    options.model,
    paths.modelPath,
    paths.pythonPath,
    paths.device,
    computeGpuIndex ?? "auto",
    paths.runRootDir,
  ].join("\n");
  const lease = await lamaEnginePool.acquire(key, () =>
    prepareLamaEngineCandidate(options),
  );
  if (lease.reused) {
    options.onProgress?.({
      progressText: "Anime/Manga LaMa ready",
      detail: "Using the cached LaMa runtime",
      progressMode: "log-only",
      installLogLine: "Anime/Manga LaMa runtime cache",
    });
  } else {
    logInpaintingRuntimeInfo("LaMa inpainting engine cached", {
      model: options.model,
      backend: paths.device,
      ttlMs: LAMA_ENGINE_IDLE_TTL_MS,
    });
  }
  return { engine: lease.resource, release: lease.release };
}

async function prepareLamaEngineCandidate(
  options: AcquireLamaEngineOptions,
): Promise<LamaInpaintingEngine> {
  let engine: LamaInpaintingEngine | null = null;
  try {
    engine = await prepareLamaInpaintingEngine({
      appPaths: options.appPaths,
      backend: options.backend,
      computeGpuIndex: options.computeGpuIndex,
      signal: options.signal,
      onProgress: options.onProgress,
    });
    await smokeTestLamaEngine(engine, options.signal);
    return engine;
  } catch (error) {
    if (engine) {
      const failedEngine = engine;
      await releaseModelResource(failedEngine, () => failedEngine.dispose());
    }
    throw error;
  }
}

async function smokeTestLamaEngine(
  engine: LamaInpaintingEngine,
  signal?: AbortSignal,
): Promise<void> {
  signal?.throwIfAborted();
  const width = 512;
  const height = 512;
  const bitmap = Buffer.alloc(width * height * 4, 255);
  const mask = new Uint8Array(width * height);
  for (let y = 160; y < 224; y += 1) {
    mask.fill(1, y * width + 160, y * width + 224);
  }
  await engine.inpaint(bitmap, width, height, mask, [
    { x: 128, y: 128, w: 128, h: 128 },
  ]);
  signal?.throwIfAborted();
}

export async function disposeCachedLamaInpaintingEngine(
  reason: string,
): Promise<boolean> {
  return lamaEnginePool.dispose(reason);
}

async function disposeLamaEngine(
  engine: LamaInpaintingEngine,
  reason: string,
): Promise<void> {
  try {
    await releaseModelResource(engine, () => engine.dispose());
    logInpaintingRuntimeInfo("LaMa inpainting engine disposed", { reason });
  } catch (error) {
    logInpaintingRuntimeError(
      "Failed to dispose cached LaMa inpainting engine",
      {
        reason,
        error,
      },
    );
    throw error;
  }
}
