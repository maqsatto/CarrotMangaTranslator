import { existsSync } from "node:fs";
import { join } from "node:path";
import type { AppPaths } from "../appPaths";
import type { KoharuInpaintingBackend } from "../../shared/inpaintingSettingsTypes";
import type {
  InpaintingEngine,
  InpaintingRuntimeProgress,
} from "./inpaintingEngine";
import type { KoharuWorkerLaunchSpec } from "./koharuWorkerTypes";
import { KoharuWorker } from "./koharuWorker";
import { runKoharuInpaint } from "./koharuEngineRunner";

const LAMA_MODEL_FILE = "anime-manga-big-lama.pt";
const LAMA_WORKER_FILE = "lama-manga-python-worker.py";

export type LamaInpaintingEngine = InpaintingEngine & {
  model: "lama-manga";
};

export type LamaEnginePaths = {
  modelPath: string;
  pythonPath: string;
  workerPath: string;
  runRootDir: string;
  device: "cuda" | "cpu";
};

export async function prepareLamaInpaintingEngine(options: {
  appPaths: AppPaths;
  backend: KoharuInpaintingBackend;
  computeGpuIndex?: number;
  signal?: AbortSignal;
  onProgress?: (progress: InpaintingRuntimeProgress) => void;
}): Promise<LamaInpaintingEngine> {
  options.signal?.throwIfAborted();
  const paths = resolveLamaEnginePaths(options.appPaths, options.backend);
  options.onProgress?.({
    progressText: "Preparing Anime/Manga LaMa",
    detail: `${paths.device === "cuda" ? "CUDA" : "CPU"} · ${LAMA_MODEL_FILE}`,
    progressMode: "log-only",
    installLogLine: "Anime/Manga LaMa Linux backend",
  });
  const launch: KoharuWorkerLaunchSpec = {
    backend: paths.device === "cuda" ? "cuda-native" : "cpu",
    computeGpuIndex: options.computeGpuIndex,
    executable: paths.pythonPath,
    runtimePath: paths.pythonPath,
    label: "Anime/Manga LaMa (Python)",
    args: [
      "-u",
      paths.workerPath,
      "--model",
      paths.modelPath,
      "--device",
      paths.device,
      "--crop-size",
      "512",
      "--context",
      "64",
    ],
    env: {
      ...process.env,
      LAMA_MODEL: paths.modelPath,
      PYTHONUNBUFFERED: "1",
      PYTORCH_CUDA_ALLOC_CONF:
        process.env.PYTORCH_CUDA_ALLOC_CONF ?? "expandable_segments:True",
    },
  };
  return createLamaEngine({
    launch,
    modelPath: paths.modelPath,
    pythonPath: paths.pythonPath,
    runRootDir: paths.runRootDir,
  });
}

function createLamaEngine(options: {
  launch: KoharuWorkerLaunchSpec;
  modelPath: string;
  pythonPath: string;
  runRootDir: string;
}): LamaInpaintingEngine {
  let worker: KoharuWorker | null = null;
  const getWorker = () => {
    if (worker && !worker.isHealthy()) {
      throw new Error(
        "LaMa worker is unhealthy; finish native cleanup before retrying the workload.",
      );
    }
    worker ??= new KoharuWorker(options.launch, {
      workerName: "LaMa 인페인팅 런타임",
    });
    return worker;
  };
  return {
    model: "lama-manga",
    runtimePath: options.pythonPath,
    modelPath: options.modelPath,
    backend: "lama-manga-python",
    runRootDir: options.runRootDir,
    isHealthy() {
      return !worker || worker.isHealthy();
    },
    async inpaint(bitmap, width, height, mask, windows, runOptions = {}) {
      await runKoharuInpaint({
        bitmap,
        getWorker,
        height,
        mask,
        runOptions,
        runRootDir: options.runRootDir,
        width,
        windows,
      });
    },
    async dispose() {
      await worker?.dispose();
      worker = null;
    },
  };
}

export function resolveLamaEnginePaths(
  appPaths: AppPaths,
  backend: KoharuInpaintingBackend,
): LamaEnginePaths {
  const modelPath = firstExisting([
    process.env.MGT_LAMA_MODEL,
    join(process.cwd(), "inpaint-runtime", "models", LAMA_MODEL_FILE),
    join(
      appPaths.dataRoot,
      "models",
      "inpainting",
      "lama-manga",
      LAMA_MODEL_FILE,
    ),
  ]);
  if (!modelPath) {
    throw new Error(
      `Anime/Manga LaMa model not found. Put ${LAMA_MODEL_FILE} in inpaint-runtime/models or set MGT_LAMA_MODEL.`,
    );
  }

  const pythonPath = firstExisting([
    process.env.MGT_LAMA_PYTHON,
    join(process.cwd(), "inpaint-runtime", ".venv-lama-cu126", "bin", "python"),
    join(
      appPaths.dataRoot,
      "runtime",
      "lama-inpainting",
      "venv",
      "bin",
      "python",
    ),
  ]);
  if (!pythonPath) {
    throw new Error(
      "LaMa Python environment not found. Create inpaint-runtime/.venv-lama-cu126 with Python 3.11 + PyTorch CUDA, or set MGT_LAMA_PYTHON.",
    );
  }

  const workerPath = firstExisting([
    join(process.resourcesPath ?? "", "app-runtime", LAMA_WORKER_FILE),
    join(process.cwd(), "out", "app-runtime", LAMA_WORKER_FILE),
    join(process.cwd(), "src", "main", "runtime", LAMA_WORKER_FILE),
  ]);
  if (!workerPath) {
    throw new Error(`LaMa Python worker not found: ${LAMA_WORKER_FILE}`);
  }

  return {
    modelPath,
    pythonPath,
    workerPath,
    runRootDir: join(appPaths.dataRoot, "tmp", "runtime", "lama-inpainting"),
    device: backend === "cpu" ? "cpu" : "cuda",
  };
}

function firstExisting(candidates: Array<string | undefined>): string | null {
  for (const candidate of candidates) {
    if (candidate && existsSync(candidate)) return candidate;
  }
  return null;
}
