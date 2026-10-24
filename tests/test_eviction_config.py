"""CPU-only validation of experiment metadata; no GPU or model downloads."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eviction_runner", ROOT / "scripts/run_eviction.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
report_spec = importlib.util.spec_from_file_location("eviction_report", ROOT / "scripts/eviction_report.py")
report = importlib.util.module_from_spec(report_spec)
report_spec.loader.exec_module(report)


class ExperimentConfigTests(unittest.TestCase):
    def test_model_families_use_matched_experiment_grids(self):
        configs = [runner.load_experiment_config(ROOT / "configs" / f"eviction_{family}.yaml")
                   for family in ("llama", "qwen", "mistral")]
        for key in ("eviction", "sweep", "needle", "ppl", "seed"):
            self.assertEqual(configs[0][key], configs[1][key])
            self.assertEqual(configs[1][key], configs[2][key])
        self.assertIn(0.05, configs[0]["eviction"]["ratios"])
        self.assertEqual(configs[0]["eviction"]["policies"], ["recent", "sink_window", "h2o"])
        self.assertEqual(len(runner.methods(configs[0], 1024)), 13)

    def test_quick_methods_include_five_percent(self):
        cfg = runner.load_experiment_config(ROOT / "configs/eviction_llama.yaml", quick=True)
        methods = runner.methods(cfg, 1024)
        self.assertEqual(len(methods), 7)
        for policy in ("recent", "sink_window", "h2o"):
            entry = next(m for m in methods if m["policy"] == policy and m["ratio"] == 0.05)
            self.assertEqual(entry["budget_tokens"], 51)
        self.assertEqual(methods[-1]["method"], "h2o_style_r0.05")

    def test_unknown_or_quantized_policy_is_rejected(self):
        import yaml
        cfg = runner.load_experiment_config(ROOT / "configs/eviction_llama.yaml")
        cfg["eviction"]["policies"] = ["int8"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(ValueError, "eviction.policies"):
                runner.load_experiment_config(path)

    def test_bfloat_reference_is_named_and_paired(self):
        cfg = {"dtype": "bfloat16", "eviction": {"ratios": [0.05]}}
        self.assertEqual(runner.methods(cfg, 1024)[0]["method"], "full_bfloat16")
        common = dict(stage="sweep", context=1024, batch=1, status="ok", trial=0)
        rows = [dict(common, method="full_bfloat16", tpot_ms=10),
                dict(common, method="recent_r0.05", tpot_ms=5)]
        self.assertEqual(report.matched_comparisons(rows)[0]["tpot_ms_ratio_to_full"], 0.5)

    def test_invalid_budget_rejected(self):
        import yaml
        cfg = runner.load_experiment_config(ROOT / "configs/eviction_llama.yaml")
        cfg["eviction"]["ratios"] = [0.001]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(ValueError, "retention budget"):
                runner.load_experiment_config(path)

    def test_resume_recovers_only_truncated_final_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            log = runner.DurableLog(path)
            log.append(dict(key="a", status="ok"))
            with path.open("ab") as f:
                f.write(b'{"key":')
            recovered = runner.DurableLog(path)
            self.assertEqual(recovered.keys, {"a"})
            self.assertTrue(path.with_suffix(".interrupted-tail").exists())
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                recovered.append(dict(key="a"))

    def test_complete_record_without_newline_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps(dict(key="a")))
            log = runner.DurableLog(path)
            log.append(dict(key="b"))
            self.assertEqual(len(runner.DurableLog(path).rows), 2)

    def test_empty_log_exists_for_report_after_expired_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            self.assertEqual(runner.DurableLog(path).rows, [])
            self.assertEqual(path.read_text(), "")

    def test_strict_scoring_rejects_explanation(self):
        self.assertEqual(runner.normalize_answer(" 7319\n"), "7319")
        self.assertNotEqual(runner.normalize_answer("The answer is 7319"), "7319")

    def test_paired_quality_excludes_failed_reference(self):
        rows = [dict(stage="needle", context=1024, batch=1, depth=0.5, sample=0,
                     method="full_fp16", status="oom"),
                dict(stage="needle", context=1024, batch=1, depth=0.5, sample=0,
                     method="recent_r0.5", status="ok", correct=False)]
        self.assertEqual(report.matched_comparisons(rows), [])

    def test_ppl_uses_token_weighting(self):
        rows = [dict(stage="ppl", context=10, batch=1, method="full_fp16", status="ok", n=2, nll=2),
                dict(stage="ppl", context=10, batch=1, method="full_fp16", status="ok", n=8, nll=16)]
        import math
        self.assertAlmostEqual(report.aggregate(rows)[0]["ppl"], math.exp(1.8))

    def test_sweep_summary_uses_median(self):
        rows = [dict(stage="sweep", context=1024, batch=1, method="full_fp16", status="ok", tpot_ms=n)
                for n in (10, 11, 90)]
        self.assertEqual(report.aggregate(rows)[0]["tpot_ms"], 11)

    def test_failure_analysis_separates_baseline_failure_and_retention(self):
        rows = []
        for sample, baseline, optimized, retained in ((0, True, False, 0.0), (1, True, False, 1.0),
                                                       (2, False, False, 0.0), (3, True, True, 0.0)):
            common = dict(stage="needle", context=1024, batch=1, depth=0.5, sample=sample,
                          status="ok", prompt_sha=str(sample))
            rows.extend([dict(common, method="full_fp16", correct=baseline),
                         dict(common, method="recent_r0.5", correct=optimized, answer_retained_fraction=retained)])
        categories = [r["category"] for r in report.failure_analysis(rows)]
        self.assertEqual(categories, ["new_failure_code_partly_or_fully_evicted", "new_failure_code_retained",
                                      "baseline_failed", "both_correct"])

    def test_depth_summary_preserves_position_and_failures(self):
        rows = [dict(stage="needle", context=1024, depth=depth, method="full_fp16", status=status,
                     correct=True) for depth, status in ((0.0, "ok"), (0.0, "oom"), (0.5, "ok"))]
        cells = report.depth_summary(rows)
        self.assertEqual(len(cells), 2)
        self.assertEqual((cells[0]["recorded"], cells[0]["successful"]), (2, 1))


if __name__ == "__main__":
    unittest.main()
