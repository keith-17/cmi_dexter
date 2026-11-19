"""Static anti-reinvention and naming-hygiene audit tests across src/."""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


class AntiReinventionAndNamingAuditTests(unittest.TestCase):
    def _iter_src_files(self):
        for p in sorted(SRC.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            yield p

    def test_no_fake_approximate_entropy_proxy_anywhere(self):
        for p in self._iter_src_files():
            text = p.read_text(encoding="utf-8")
            self.assertNotIn("Fast bounded regularity proxy", text, f"Fake ApEn proxy found in {p}")
            self.assertNotIn("log1p(mean_abs_change", text, f"Fake ApEn formula found in {p}")

    def test_tsaug_isolated_inside_base_utils_adapter(self):
        for p in self._iter_src_files():
            if p.name == "base_utils.py":
                continue
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotEqual(alias.name, "tsaug", f"Direct `import tsaug` outside base_utils TsaugAdapter in {p}")
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotEqual(node.module, "tsaug", f"Direct `from tsaug import` outside base_utils TsaugAdapter in {p}")

    def test_no_manual_quaternion_3x3_matrix_formulas_in_refactored_code(self):
        # Verify manual Euler / 3x3 rotation matrix formulas are delegated to scipy.spatial.transform.Rotation
        targets = [
            SRC / "base_utils.py",
            SRC / "utils.py",
        ]
        manual_pattern = re.compile(r"1\s*-\s*2\s*\*\s*\(\s*y\*\*2\s*\+\s*z\*\*2\s*\)")
        for p in targets:
            text = p.read_text(encoding="utf-8")
            self.assertIsNone(manual_pattern.search(text), f"Manual 3x3 rotation matrix found in {p}")

    def test_consolidated_two_file_layout_removes_scattered_packages(self):
        self.assertTrue((SRC / "base_utils.py").is_file())
        self.assertTrue((SRC / "utils.py").is_file())
        for removed in ("utils_v3.py", "search_drivers.py", "features", "signal_descriptors", "augmentation"):
            self.assertFalse((SRC / removed).exists(), f"Scattered legacy artifact still exists: src/{removed}")

    def test_zero_chatbot_names_in_src(self):
        # Match chatbot names while ignoring mathematical t-SNE ``perplexity=`` parameter
        forbidden = re.compile(
            r"\b(claude|mistral|perplexity\s+ai|openai|chatgpt|grok|deepseek|qwen|meta\s+ai)\b",
            re.IGNORECASE,
        )
        for p in self._iter_src_files():
            text = p.read_text(encoding="utf-8")
            match = forbidden.search(text)
            if match is not None:
                self.fail(f"Forbidden chatbot reference {match.group(0)!r} found in {p}")


if __name__ == "__main__":
    unittest.main()
