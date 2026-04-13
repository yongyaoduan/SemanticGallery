from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from deployment.public_anchor import normalize_public_anchor_extract


class PublicAnchorTests(unittest.TestCase):
    def test_normalize_public_anchor_extract_rewrites_stale_screen2words_paths(self):
        with tempfile.TemporaryDirectory(prefix="sg-public-anchor-") as tmp_dir:
            extract_root = Path(tmp_dir) / "extracted"
            image_path = extract_root / "screen2words" / "images" / "rico" / "66545.jpg"
            manifest_path = extract_root / "screen2words" / "manifest.jsonl"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"jpg")
            manifest_path.write_text(
                json.dumps(
                    {
                        "image_path": "/private/tmp/sg_public_anchor_seeded/screen2words/images/rico/66545.jpg",
                        "captions": ["recipe app"],
                        "split": "train",
                        "source": "screen2words",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            repaired_rows = normalize_public_anchor_extract(extract_root)

            self.assertEqual(repaired_rows, 1)
            repaired_row = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(repaired_row["image_path"], "images/rico/66545.jpg")


if __name__ == "__main__":
    unittest.main()
