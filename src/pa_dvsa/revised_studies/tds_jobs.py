"""Parallel time-domain simulation jobs, each on a fresh copy of the system."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
import logging
import math
import multiprocessing as mp
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from . import config as C

_CASES: "OrderedDict[tuple[str, str], Any]" = OrderedDict()
_MAX_CACHED = 2


@contextmanager
def quiet() -> Iterator[None]:
    logger = logging.getLogger("andes")
    old = logger.level
    logger.setLevel(logging.ERROR)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        logger.setLevel(old)


def case_fingerprint(case: Any) -> str:
    payload = {
        "slug": case.slug,
        "variant": case.variant,
        "blocks": [
            [b.asset_id, str(b.collector_bus), str(b.hv_bus), round(float(b.tap), 12), round(float(b.z0_pu), 12),
             round(float(b.p_mw), 9), float(b.pickup_pu), float(b.dwell_s)]
            for b in case.blocks
        ],
        "events": [[e.event_id, [[m.model, str(m.device_idx), m.attr, round(float(m.value), 12)] for m in e.mutations]]
                   for e in case.events],
        "controls": [[c.control_id, [[m.model, str(m.device_idx), m.attr, round(float(m.value), 12)] for m in c.mutations]]
                     for c in case.controls],
        "relays_enabled": bool(case.relays_enabled),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def register_case(case: Any) -> None:
    _CASES[(case.slug, case.variant)] = case
    _CASES.move_to_end((case.slug, case.variant))


def get_case(slug: str, variant: str) -> Any:
    key = (slug, variant)
    if key in _CASES:
        _CASES.move_to_end(key)
        return _CASES[key]
    from ..benchmarks import build_benchmark

    with quiet():
        case = build_benchmark(slug, variant=variant)
    _CASES[key] = case
    while len(_CASES) > _MAX_CACHED:
        _CASES.popitem(last=False)
    return case


def _all_bus_addresses(case: Any) -> dict[str, int]:
    ss = case.system
    return {f"V_{b}": int(a) for b, a in zip(ss.Bus.idx.v, ss.Bus.v.a)}


def tds_job(job: Mapping[str, Any]) -> dict[str, Any]:
    t_job = time.perf_counter()
    out: dict[str, Any] = {"job_id": job["job_id"], "kind": job.get("kind"), "label": job.get("label")}
    try:
        from ..tds_runner import run_protection_tds

        case = get_case(job["slug"], job["variant"])
        fp = case_fingerprint(case)
        if job.get("fingerprint") and fp != job["fingerprint"]:
            raise RuntimeError(f"case fingerprint mismatch in worker ({fp} != {job['fingerprint']})")
        seed = case.event(job["seed_id"]) if job.get("seed_id") else None
        controls = [(case.control(cid), float(alpha)) for cid, alpha in job.get("controls", ())]
        build = {"s": float("nan")}

        def factory() -> Any:
            t = time.perf_counter()
            with quiet():
                ss = case.factory()
            build["s"] = time.perf_counter() - t
            return ss

        extra = _all_bus_addresses(case) if job.get("all_bus_dv") else None
        relays = case.relays_enabled if job.get("relays_enabled") is None else bool(job["relays_enabled"])
        with quiet():
            res = run_protection_tds(
                factory,
                seed,
                case.blocks,
                relays_enabled=relays,
                controls=controls,
                trip_events={e.event_id: e for e in case.events},
                t_seed=C.T_SEED_S,
                window_s=float(job.get("window_s", C.T_WINDOW_S)),
                t_end=float(job.get("t_end", C.TDS_T_END_S)),
                tstep=float(job.get("tstep", C.TDS_TSTEP_S)),
                record_traces=True,
                trace_addresses=extra,
            )
        d = res.to_dict()
        out.update(d)
        out["build_s"] = build["s"]
        out["sim_s"] = float(res.runtime_s - build["s"]) if math.isfinite(build["s"]) else float("nan")
        tr = res.traces or {}
        t = np.asarray(tr.get("t", []), dtype=float)
        v = np.asarray(tr.get("v_collector", np.zeros((0, 0))), dtype=float)
        out["trace_t"] = t.tolist()
        out["trace_v_collector"] = v.tolist()
        out["trace_asset_ids"] = [str(a) for a in tr.get("asset_ids", [])]
        if extra is not None and "extra" in tr and t.size:
            ev = np.asarray(tr["extra"], dtype=float)
            k_pre = int(np.flatnonzero(np.isclose(t, C.T_SEED_S, atol=1e-9))[0]) if np.any(np.isclose(t, C.T_SEED_S, atol=1e-9)) else int(np.searchsorted(t, C.T_SEED_S) - 1)
            post = ev[k_pre + 1:, :]
            dv = np.abs(post - ev[k_pre, :]) if post.size else np.zeros((0, ev.shape[1]))
            per_bus = dv.max(axis=0) if dv.size else np.zeros(ev.shape[1])
            ids = [str(x) for x in tr["extra_ids"]]
            kmax = int(np.argmax(per_bus)) if per_bus.size else 0
            out["max_abs_dv_all_buses_pu"] = float(per_bus.max()) if per_bus.size else 0.0
            out["max_abs_dv_bus"] = ids[kmax] if ids else None
            out["n_buses_moved_1e-4"] = int(np.sum(per_bus > C.NOOP_DV_PU))
        out["error"] = None
    except Exception as exc:
        out.update({"status": "job_error", "converged": False, "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()})
    out["job_wall_s"] = time.perf_counter() - t_job
    out["job"] = {k: v for k, v in job.items()}
    return out


def make_job(
    job_id: str,
    case: Any,
    seed_id: str | None,
    *,
    kind: str,
    controls: Sequence[tuple[str, float]] = (),
    label: str = "",
    fingerprint: str | None = None,
    tstep: float = C.TDS_TSTEP_S,
    all_bus_dv: bool = False,
    relays_enabled: bool | None = None,
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "slug": case.slug,
        "variant": case.variant,
        "fingerprint": fingerprint or case_fingerprint(case),
        "seed_id": seed_id,
        "controls": [(str(c), float(a)) for c, a in controls],
        "kind": kind,
        "label": label,
        "tstep": float(tstep),
        "all_bus_dv": bool(all_bus_dv),
        "relays_enabled": relays_enabled,
        "window_s": float(C.T_WINDOW_S),
        "t_end": float(C.TDS_T_END_S),
    }


def _worker_init(fault_dir: str | None = None) -> None:
    warnings.simplefilter("ignore")
    logging.getLogger("andes").setLevel(logging.ERROR)
    if fault_dir:
        import faulthandler
        import os
        from pathlib import Path

        Path(fault_dir).mkdir(parents=True, exist_ok=True)
        fh = open(Path(fault_dir) / f"worker_{os.getpid()}.log", "w")
        faulthandler.enable(file=fh, all_threads=True)


class TDSPool:
    def __init__(self, n_jobs: int, fault_dir: str | None = None) -> None:
        self.n_jobs = max(1, min(int(n_jobs), 6))
        self.fault_dir = fault_dir
        self._pool: ProcessPoolExecutor | None = None
        if self.n_jobs > 1:
            self._start()

    def _start(self) -> None:
        self._pool = ProcessPoolExecutor(
            max_workers=self.n_jobs, mp_context=mp.get_context("spawn"), initializer=_worker_init,
            initargs=(self.fault_dir,),
        )

    def submit_all(self, jobs: Sequence[Mapping[str, Any]]) -> list[Any]:
        if self._pool is None:
            return [("inline", j) for j in jobs]
        try:
            return [(self._pool.submit(tds_job, j), j) for j in jobs]
        except Exception:
            self._restart()
            return [(self._pool.submit(tds_job, j), j) for j in jobs]

    def _restart(self) -> None:
        try:
            if self._pool is not None:
                self._pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self.restarts = getattr(self, "restarts", 0) + 1
        self._start()

    def gather(self, handles: Sequence[Any], log: Callable[[str], None] | None = None,
               max_retries: int = 2) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        pending = list(handles)
        for attempt in range(max_retries + 1):
            retry: list[Mapping[str, Any]] = []
            futs = {}
            for h in pending:
                if h[0] == "inline":
                    r = tds_job(h[1])
                    results[r["job_id"]] = r
                else:
                    futs[h[0]] = h[1]
            for f in as_completed(list(futs)):
                job = futs[f]
                try:
                    r = f.result()
                except Exception as exc:
                    retry.append(job)
                    if log:
                        log(f"    worker failure on {job['job_id']}: {type(exc).__name__}: {exc}")
                    continue
                results[r["job_id"]] = r
                if log and (r.get("error") or len(results) % 25 == 0):
                    log(f"    TDS {len(results)} done; last {r['job_id']}: {r.get('status')} {r.get('error') or ''}")
            if not retry:
                break
            if attempt == max_retries:
                for job in retry:
                    r = tds_job(job)
                    r["note"] = "run inline after repeated worker failures"
                    results[r["job_id"]] = r
                break
            self._restart()
            pending = [(self._pool.submit(tds_job, j), j) for j in retry]
            for j in retry:
                results.pop(j["job_id"], None)
        self.retried = getattr(self, "retried", 0)
        return results

    def run(self, jobs: Sequence[Mapping[str, Any]], log: Callable[[str], None] | None = None) -> dict[str, dict[str, Any]]:
        return self.gather(self.submit_all(jobs), log)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=False)
            self._pool = None
