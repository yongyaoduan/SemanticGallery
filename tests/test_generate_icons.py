from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class GenerateIconsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module(Path("desktop/scripts/generate_icons.py"), "generate_icons")

    def test_render_icon_returns_expected_size_and_non_empty_pixels(self):
        icon = self.module.render_icon(256)
        self.assertEqual(icon.size, (256, 256))
        self.assertGreater(icon.getbbox()[2], 0)

    def test_build_icons_writes_required_bundle_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            written = self.module.build_icons(Path(temp_dir))
            names = {path.name for path in written}
            self.assertEqual(names, {"icon.png", "128x128.png", "128x128@2x.png", "32x32.png", "icon.icns"})
            with Image.open(Path(temp_dir) / "icon.png") as image:
                self.assertEqual(image.size, (1024, 1024))


if __name__ == "__main__":
    unittest.main()
