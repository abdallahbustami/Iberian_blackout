from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pa_dvsa.environment import check_environment
from pa_dvsa.manifest import collect_manifest
from pa_dvsa.phase0 import main as phase0_main
from pa_dvsa.traceability import extract_traceability


class TraceabilityTests(unittest.TestCase):
    def test_root_tex_traceability_contains_core_paper_artifacts(self) -> None:
        payload = extract_traceability(PROJECT_ROOT / "LaTeX/root.tex")
        labels = {item["label"] for item in payload["items"]}

        expected = {
            "eq:hybrid_dae",
            "eq:reduced_matrices",
            "eq:K_exact",
            "eq:mitigation_qp",
            "alg:finite_window_maps",
            "alg:operational_screen",
            "fig:cascade_lift",
            "prop:fixed_point",
            "rem:proxy_use",
        }
        self.assertTrue(expected.issubset(labels))
        self.assertGreater(payload["source"]["line_count"], 1100)
        self.assertGreaterEqual(payload["summary"]["required_outputs"], 6)

        section_items = {item["label"]: item for item in payload["items"] if item["kind"] == "section"}
        self.assertEqual(section_items["subsec:rootcause_summary"]["implementation_status"], "reference")
        self.assertEqual(
            section_items["subsec:rootcause_summary"]["title"],
            "Root cause structure for the screening problem",
        )

    def test_required_outputs_include_validation_and_mitigation(self) -> None:
        payload = extract_traceability(PROJECT_ROOT / "LaTeX/root.tex")
        output_ids = {item["output_id"] for item in payload["required_outputs"]}
        self.assertIn("screen_vs_nonlinear_heatmap", output_ids)
        self.assertIn("mitigation_lp_qp_results", output_ids)


class EnvironmentTests(unittest.TestCase):
    def test_environment_check_reports_dependencies_and_inputs(self) -> None:
        report = check_environment(PROJECT_ROOT)
        dependency_names = {dep["name"] for dep in report["dependencies"]}
        input_paths = {item["path"] for item in report["inputs"]}

        self.assertIn("andes", dependency_names)
        self.assertIn("numpy", dependency_names)
        self.assertIn("LaTeX/root.tex", input_paths)
        self.assertIn("docs-andes-app-en-stable.pdf", input_paths)
        self.assertIn("ready_for_andes_screening", report)


class ManifestTests(unittest.TestCase):
    def test_manifest_records_input_checksums_without_git_requirement(self) -> None:
        manifest = collect_manifest(PROJECT_ROOT, "unit-test", input_paths=["LaTeX/root.tex"])
        self.assertEqual(manifest["run_name"], "unit-test")
        self.assertEqual(len(manifest["inputs"]), 1)
        self.assertTrue(manifest["inputs"][0]["exists"])
        self.assertEqual(len(manifest["inputs"][0]["sha256"]), 64)
        self.assertIn("is_repository", manifest["git"])

    def test_phase0_cli_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = phase0_main(
                [
                    "--project-root",
                    str(PROJECT_ROOT),
                    "--out-dir",
                    tmp,
                ]
            )
            out = Path(tmp)
            self.assertEqual(code, 0)
            self.assertTrue((out / "traceability.json").exists())
            self.assertTrue((out / "environment.json").exists())
            self.assertTrue((out / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
