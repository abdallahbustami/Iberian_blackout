"""Tests of the scoring, ranking and observability helpers, and a small end-to-end run on Kundur."""

from __future__ import annotations

import math
import unittest

from pa_dvsa.revised_studies import scoring as SC
from pa_dvsa.revised_studies import triage as TR


def _screen(**classes):
    out = {}
    for a, spec in classes.items():
        cls, trip, t, layer = (list(spec) + [None, None, None])[:4]
        out[a] = {"asset_class": cls, "predicted_trip": bool(trip), "predicted_time_s": t, "layer": layer, "score": 1.0}
    return out


class ScoringRuleTests(unittest.TestCase):
    def test_labels_and_outcome_table(self) -> None:
        self.assertEqual(SC.screen_label("pickup_predicted", True), "trip")
        self.assertEqual(SC.screen_label("pickup_predicted", False), "pickup_only")
        self.assertEqual(SC.screen_label("risky", False), "risky")
        with self.assertRaises(SC.ScoringError):
            SC.screen_label("certified", True)
        self.assertEqual(SC.tds_label(True, None, 1.0), "no_trip")
        self.assertEqual(SC.tds_label(True, 0.2, 1.0), "trip")
        self.assertEqual(SC.tds_label(True, 1.0, 1.0), "trip")
        self.assertEqual(SC.tds_label(True, 1.2, 1.0), "late_trip")
        self.assertEqual(SC.tds_label(False, 0.2, 1.0), "inconclusive")
        table = {
            ("trip", "trip"): "TP", ("trip", "no_trip"): "FP", ("pickup_only", "trip"): "pickup_only_trip",
            ("risky", "trip"): "caught_risky", ("risky", "no_trip"): "risky_clear",
            ("certified", "trip"): "unsafe_FN", ("certified", "no_trip"): "TN",
            ("data_limited", "trip"): "data_limited_trip", ("certified", "late_trip"): "late_trip",
            ("trip", "inconclusive"): "tds_inconclusive",
        }
        for (s, t), o in table.items():
            self.assertEqual(SC.outcome(s, t), o)

    def test_seed_scoring_counts_sets_and_timing(self) -> None:
        screen = _screen(A=("pickup_predicted", True, 0.06, 1), B=("pickup_predicted", True, 0.02, 2),
                         C=("certified",), D=("risky",), E=("certified",), F=("pickup_predicted", False, None, 1))
        tds = {"A": {"trip_rel_s": 0.07}, "B": {"trip_rel_s": None}, "C": {"trip_rel_s": 0.5},
               "D": {"trip_rel_s": 0.3}, "E": {"trip_rel_s": 1.3}, "F": {"trip_rel_s": None},
               "S": {"trip_rel_s": None}}
        rows, summ = SC.score_seed(seed_id="s", family="f", self_assets=("S",), screen=screen, tds_assets=tds,
                                   tds_ok=True, window_s=1.0)
        out = {r["asset_id"]: r["outcome"] for r in rows}
        self.assertEqual(out, {"A": "TP", "B": "FP", "C": "unsafe_FN", "D": "caught_risky", "E": "late_trip",
                               "F": "pickup_only_clear"})
        self.assertNotIn("S", out)
        self.assertEqual((summ["n_TP"], summ["n_FP"], summ["n_unsafe_FN"], summ["n_caught_risky"], summ["n_late_trip"]),
                         (1, 1, 1, 1, 1))
        self.assertEqual(summ["n_assets_scored"], 5)
        self.assertTrue(summ["dangerous"])
        self.assertEqual(summ["predicted_set"], ["A", "B"])
        self.assertEqual(summ["tds_set"], ["A", "C", "D"])
        self.assertAlmostEqual(summ["jaccard"], 1 / 4)
        self.assertEqual(summ["pred_first_asset"], "A")
        self.assertEqual(summ["tds_first_asset"], "A")
        self.assertAlmostEqual(summ["timing_error_s"], 0.06 - 0.07)
        self.assertEqual((summ["n_TP_L1"], summ["n_FP_L2p"]), (1, 1))
        self.assertFalse(summ["seed_unsafe_FN"])

    def test_self_term_in_screen_is_rejected(self) -> None:
        with self.assertRaises(SC.ScoringError):
            SC.score_seed(seed_id="s", family="f", self_assets=("A",), screen=_screen(A=("certified",)),
                          tds_assets={}, tds_ok=True, window_s=1.0)

    def test_certified_seed_with_trip_is_a_seed_level_unsafe_fn(self) -> None:
        rows, summ = SC.score_seed(seed_id="s", family="f", self_assets=(), screen=_screen(A=("certified",), B=("certified",)),
                                   tds_assets={"A": {"trip_rel_s": 0.4}}, tds_ok=True, window_s=1.0)
        self.assertTrue(summ["certified_no_secondary"])
        self.assertTrue(summ["seed_unsafe_FN"])
        self.assertEqual(summ["n_unsafe_FN"], 1)
        self.assertEqual(summ["n_TN"], 1)

    def test_both_empty_and_inconclusive(self) -> None:
        _, summ = SC.score_seed(seed_id="s", family="f", self_assets=(), screen=_screen(A=("certified",)),
                                tds_assets={"A": {"trip_rel_s": None}}, tds_ok=True, window_s=1.0)
        self.assertEqual((summ["jaccard"], summ["both_empty"], summ["dangerous"]), (1.0, True, False))
        rows, summ = SC.score_seed(seed_id="s", family="f", self_assets=(), screen=_screen(A=("pickup_predicted", True, 0.1, 1)),
                                   tds_assets={"A": {"trip_rel_s": 0.1}}, tds_ok=False, window_s=1.0)
        self.assertEqual(rows[0]["outcome"], "tds_inconclusive")
        self.assertIsNone(summ["dangerous"])
        self.assertTrue(math.isnan(summ["jaccard"]))
        self.assertEqual(summ["n_assets_scored"], 0)

    def test_aggregate(self) -> None:
        s1 = SC.score_seed(seed_id="a", family="f", self_assets=(), screen=_screen(A=("pickup_predicted", True, 0.1, 1)),
                           tds_assets={"A": {"trip_rel_s": 0.12}}, tds_ok=True, window_s=1.0)[1]
        s2 = SC.score_seed(seed_id="b", family="f", self_assets=(), screen=_screen(A=("certified",)),
                           tds_assets={"A": {"trip_rel_s": None}}, tds_ok=True, window_s=1.0)[1]
        s3 = SC.score_seed(seed_id="c", family="f", self_assets=(), screen=_screen(A=("certified",)),
                           tds_assets={}, tds_ok=False, window_s=1.0)[1]
        agg = SC.aggregate([s1, s2, s3], group={"family": "f"})
        self.assertEqual((agg["n_seeds"], agg["n_seeds_inconclusive"], agg["n_dangerous"]), (3, 1, 1))
        self.assertEqual((agg["TP"], agg["TN"], agg["tds_inconclusive"], agg["seed_TP"], agg["seed_TN"]), (1, 1, 1, 1, 1))
        self.assertAlmostEqual(agg["jaccard_mean_nonempty"], 1.0)
        self.assertEqual(agg["n_both_empty"], 1)
        self.assertAlmostEqual(agg["timing_abs_error_max_s"], 0.02)


class TriageTests(unittest.TestCase):
    def test_capture_curve_and_auc(self) -> None:
        scores = {"a": (3.0, 1.0), "b": (2.0, 0.5), "c": (2.0, 0.1), "d": (0.1, 0.0), "e": (float("nan"), float("nan"))}
        labels = {"a": True, "b": False, "c": True, "d": False, "e": None}
        order = list(scores)
        ranked = TR.rank_seeds(scores, order)
        self.assertEqual(ranked, ["a", "b", "c", "d", "e"])
        curve = TR.capture_curve(ranked, labels)
        self.assertEqual([r["fraction_captured"] for r in curve], [0.0, 0.5, 0.5, 1.0, 1.0])
        self.assertAlmostEqual(TR.fraction_to_capture_all(curve), 0.75)
        self.assertAlmostEqual(TR.roc_auc(scores, labels), 3 / 4)
        self.assertTrue(math.isnan(TR.roc_auc(scores, {"a": True})))


class DropBoundTests(unittest.TestCase):
    def test_absorbing_fixed_pf_plant_has_nonpositive_drop(self) -> None:
        from types import SimpleNamespace

        from pa_dvsa.revised_studies.observability import gsu_drop_interval

        hid = SimpleNamespace(gsu_r_pu_sysbase=0.0016, gsu_x_pu_sysbase=0.066, drop_bound_ilim_pu=0.156,
                              drop_op_pu=-0.034, drop_bound_op_pu=0.10)
        d = gsu_drop_interval(hid, SimpleNamespace(q_mvar=-52.0, p_mw=158.0))
        self.assertEqual((d["drop_lower_pu"], d["drop_upper_pu"]), (-0.156, 0.0))
        self.assertLessEqual(d["drop_op_pu"], d["drop_upper_pu"])
        d2 = gsu_drop_interval(hid, SimpleNamespace(q_mvar=+30.0, p_mw=158.0))
        self.assertEqual(d2["drop_upper_pu"], 0.156)

    def test_exact_gsu_relation_respects_the_sign_bound(self) -> None:
        r, x = 0.0016, 0.066
        for p, q, vc in ((1.58, -0.52, 1.07), (0.3, -0.1, 1.12), (2.2, -0.9, 1.0)):
            vhv = complex(vc, 0.0) - complex(r, x) * complex(p, -q) / vc
            drop = vc - abs(vhv)
            self.assertLessEqual(drop, (r * p + x * q) / vc + 1e-15)
            self.assertLessEqual(drop, 0.0)


    def test_gsu_reconstruction_from_hv_measurements_is_exact(self) -> None:
        import cmath

        from pa_dvsa.revised_studies.observability import gsu_reconstruct

        r, x = 0.0016, 0.066
        for p, q, vc, ang, n in ((1.58, -0.52, 1.07, 0.1, 0.95), (0.3, -0.1, 1.12, -0.2, 0.92), (2.2, -0.9, 1.0, 0.0, 1.03)):
            v_col = cmath.rect(vc, ang)
            i = (complex(p, q) / v_col).conjugate()
            v_int = v_col - complex(r, x) * i
            s_hv = v_int * i.conjugate()
            v_hv = abs(v_int) * n
            self.assertAlmostEqual(gsu_reconstruct(v_hv, s_hv.real, s_hv.imag, r, x, n), vc, places=12)


class KundurSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from pa_dvsa.revised_studies.screen import predict_all, run_screen
        from pa_dvsa.revised_studies.tds_jobs import get_case, quiet

        cls.case = get_case("kundur", "baseline")
        with quiet():
            cls.scr = run_screen(cls.case)
        cls.preds = predict_all(cls.scr)

    def test_measured_pq_regime(self) -> None:
        from pa_dvsa.revised_studies import observability as O

        models, theta, info = O.regime_setup(self.scr, self.case, "measured_pq")
        for a in self.scr.asset_ids:
            lo, hi = theta.reconstruction_error_pu[a]
            self.assertLess(lo, 0.0)
            self.assertGreater(hi, 0.0)
            self.assertLess(hi - lo, 0.02)
        b, *_ = O.regime_bounds(self.scr, self.case, "measured_pq")
        self.assertEqual(sum(1 for x in b.assets if x.data_limited), 0)
        b_kt, *_ = O.regime_bounds(self.scr, self.case, "known_tap")
        self.assertGreater(sum(1 for x in b_kt.assets if x.data_limited), 0)

    def test_screen_maps_and_self_terms(self) -> None:
        maps = self.scr.maps
        self.assertEqual(set(self.scr.failed_events), set())
        for e in self.case.events:
            j = maps.column(e.event_id)
            for i, a in enumerate(maps.asset_ids):
                self.assertEqual(maps.classes[i, j] == "self", a in e.disconnects_assets)
            p = self.preds[e.event_id]
            self.assertFalse(set(p.classes) & set(e.disconnects_assets))
            for c in p.classes.values():
                if c["predicted_trip"]:
                    self.assertEqual(c["asset_class"], "pickup_predicted")
                    self.assertTrue(c["trip_possible"])

    def test_one_tds_run_is_scored(self) -> None:
        from pa_dvsa.revised_studies import config as C
        from pa_dvsa.revised_studies.tds_jobs import make_job, tds_job

        seed = max(self.preds, key=lambda e: self.preds[e].max_k_pk_nonself)
        r = tds_job(make_job("smoke", self.case, seed, kind="smoke", all_bus_dv=True))
        self.assertIsNone(r["error"])
        self.assertEqual(r["status"], "ok")
        self.assertGreater(r["max_abs_dv_all_buses_pu"], C.NOOP_DV_PU)
        ev = self.case.event(seed)
        rows, summ = SC.score_seed(seed_id=seed, family=ev.family, self_assets=ev.disconnects_assets,
                                   screen=self.preds[seed].classes, tds_assets={a["asset_id"]: a for a in r["assets"]},
                                   tds_ok=True, window_s=C.T_WINDOW_S)
        self.assertEqual(len(rows), len(self.case.blocks) - len(ev.disconnects_assets))
        self.assertEqual(summ["n_tds_inconclusive"], 0)


if __name__ == "__main__":
    unittest.main()
