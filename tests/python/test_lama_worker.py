"""Run with the isolated LaMa Python; no GPU or model is needed."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "lama_worker", ROOT / "src/main/runtime/lama-manga-python-worker.py"
)
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


class LamaWorkerTest(unittest.TestCase):
    def test_bounded_tiles_preserve_alpha_and_unmasked_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = np.full((1600, 900, 4), 123, dtype=np.uint8)
            source[:, :, 3] = 187
            mask = np.zeros((1600, 900), dtype=np.uint8)
            mask[200:220, 200:220] = 255
            mask[300:320, 500:520] = 255
            Image.fromarray(source).save(root / "input.png")
            Image.fromarray(mask).save(root / "mask.png")
            sizes = []

            def model(image, _mask):
                sizes.append(image.size)
                return Image.new("RGB", image.size, (0, 0, 0))

            request = dict(
                input=str(root / "input.png"), mask=str(root / "mask.png"),
                output=str(root / "output.png"),
                windows=[[190, 190, 230, 230], [490, 290, 530, 330]],
            )
            WORKER.inpaint_request(model, request, 512, 64)
            result = np.asarray(Image.open(root / "output.png"))
            np.testing.assert_array_equal(result[mask == 0], source[mask == 0])
            self.assertTrue(np.all(result[mask > 0, :3] == 0))
            np.testing.assert_array_equal(result[:, :, 3], source[:, :, 3])

            # A full-height mask still uses bounded, original-resolution tiles.
            mask[:] = 255
            Image.fromarray(mask).save(root / "mask.png")
            request["windows"] = [[0, 0, 900, 1600]]
            sizes.clear()
            WORKER.inpaint_request(model, request, 512, 64)
            self.assertTrue(all(max(size) <= 512 for size in sizes))
            result = np.asarray(Image.open(root / "output.png"))
            self.assertTrue(np.all(result[:, :, :3] == 0))


if __name__ == "__main__":
    unittest.main()
