from __future__ import annotations

import unittest
from pathlib import Path


class DocsSmokeTests(unittest.TestCase):
    def test_readme_mentions_per_gallery_artifacts(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("datasets/private_gallery_local/<gallery-key>/full_manifest.jsonl", readme)
        self.assertIn("deployment/search_configs/<gallery-key>.json", readme)
        self.assertIn("Private images do not leave the machine.", readme)

    def test_privacy_docs_include_incremental_state_files(self):
        privacy = Path("docs/privacy.md").read_text(encoding="utf-8")
        self.assertIn("deployment/<gallery-key>_mlx_siglip2_file_state.json", privacy)
        self.assertIn("deployment/<gallery-key>_mlx_siglip2_bank_state.json", privacy)
        self.assertIn("logs/semanticgallery_private_data_adapted/<gallery-key>/quickstart_state.json", privacy)

    def test_reference_docs_do_not_use_legacy_config_default(self):
        reference = Path("docs/reference.md").read_text(encoding="utf-8")
        self.assertNotIn("search_config_gallery_mlx.json", reference)
        self.assertIn("deployment/search_configs/<gallery-key>.json", reference)


if __name__ == "__main__":
    unittest.main()
