#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark client for comparing solver backends on identical scrambles.

This script:
- Generates a reproducible catalog of scrambles (as color inputs)
- Sends the exact same instances to each backend via HTTP
- Aggregates results and renders summary plots
"""

import argparse
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from oc_rubik_s_cube_dqn.cube import RubiksCubeEnv
from tqdm import tqdm

BACKENDS: List[str] = [
    "cube-solver",
    "deepcube-solver",
    "dqn-solver",
    "magic-solver",
    "muzero-solver",
]


def setup_logging(verbosity: int) -> None:
    """Configure global logging level and format.

    Parameters
    ----------
    verbosity : int
        0 = warnings only, 1 = info, 2+ = debug.
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def fingerprint_color_input(color_input: Any) -> str:
    """Stable fingerprint for a color_input (identifies same scramble).

    Parameters
    ----------
    color_input : Any
        JSON-serializable structure describing the cube facelet colors.

    Returns
    -------
    str
        Short SHA-256 hex digest (12 chars) of a canonicalized representation.
    """
    try:
        canon = json.dumps(color_input, sort_keys=True, separators=(",", ":"))
    except TypeError:
        # Fallback when the object is not JSON-serializable
        canon = repr(color_input)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]


def safe_len_of_json(obj: Any) -> int:
    """Best-effort length for JSON-like objects (None -> 0)."""
    if obj is None:
        return 0
    if isinstance(obj, (list, dict, tuple, set, str)):
        return len(obj)
    return 1


def set_global_seeds(seed: int) -> None:
    """Attempt to seed various RNGs for reproducibility."""
    try:
        import random

        random.seed(seed)
    except Exception:
        pass
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def make_scramble_catalog(
    scramble_min: int,
    scramble_max: int,
    trials: int,
    base_seed: int,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """Pre-generate the scramble catalog shared across all backends.

    For each pair ``(scramble_moves, trial)`` this function creates a
    deterministic color_input using a derived seed, ensuring all backends
    receive the exact same instances.

    Parameters
    ----------
    scramble_min : int
        Minimum scramble length (inclusive).
    scramble_max : int
        Maximum scramble length (inclusive).
    trials : int
        Number of independent trials per scramble length.
    base_seed : int
        Base seed; each instance gets a derived, unique seed.
    show_progress : bool, default True
        If True, show tqdm progress bars.

    Returns
    -------
    list of dict
        Catalog rows with keys: ``scramble_moves``, ``trial``, ``color_input``,
        ``scramble_id``, ``seed``.
    """
    total_instances = (scramble_max - scramble_min + 1) * trials
    catalog: List[Dict[str, Any]] = []

    iterator = range(scramble_min, scramble_max + 1)
    if show_progress:
        iter_scrambles = tqdm(iterator, desc="Génération des scrambles", unit="scr")
    else:
        iter_scrambles = iterator

    for scr in iter_scrambles:
        trial_iter = range(1, trials + 1)

        if show_progress:
            trial_iter = tqdm(
                trial_iter, desc=f"Scramble={scr}", leave=False, unit="trial"
            )

        for t in trial_iter:
            # Instance-specific seed for stability
            seed = base_seed * 100000 + scr * 1000 + t
            set_global_seeds(seed)

            # Ideally the env accepts a seed; we derive a reproducible scramble
            env = RubiksCubeEnv(scramble_moves=scr)
            env.reset()
            color_input = env.get_colors_from_cube()
            scr_id = fingerprint_color_input(color_input)

            catalog.append(
                {
                    "scramble_moves": scr,
                    "trial": t,
                    "color_input": color_input,
                    "scramble_id": scr_id,
                    "seed": seed,
                }
            )
    assert len(catalog) == total_instances, "Catalogue incomplet."
    return catalog


def benchmark_once_on_input(
    base_url: str,
    backend: str,
    color_input: Any,
    timeout: float,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Send a single POST request to one backend for a given input.

    Parameters
    ----------
    base_url : str
        Endpoint URL (expects POST with JSON payload).
    backend : str
        Backend identifier (sent as ``X-Backend`` header).
    color_input : Any
        JSON payload representing the cube colors.
    timeout : float
        Request timeout in seconds.
    extra_headers : dict[str, str] or None, default None
        Additional headers to include in the request.

    Returns
    -------
    dict
        Result dictionary with keys: ``status_code``, ``ok``, ``json_len``,
        ``latency_ms``, ``error``.
    """
    headers = {"x_backend": backend}
    if extra_headers:
        headers.update(extra_headers)

    t0 = time.perf_counter()
    status_code = None
    resp_time_ms = None
    json_len = 0
    ok = False
    error = None

    try:
        resp = requests.post(
            base_url,
            headers=headers,
            json=color_input,
            timeout=timeout,
        )
        resp_time_ms = (time.perf_counter() - t0) * 1000.0
        status_code = resp.status_code

        if status_code == 200:
            try:
                payload = resp.json()
            except json.JSONDecodeError as e:
                error = f"JSONDecodeError: {e}"
                payload = None
            json_len = safe_len_of_json(payload)
            ok = json_len > 0
        else:
            error = f"HTTP {status_code}"

    except requests.Timeout:
        error = f"Timeout({timeout}s)"
        resp_time_ms = timeout * 1000
    except requests.RequestException as e:
        error = f"RequestException: {e}"

    return {
        "status_code": status_code,
        "ok": ok,
        "json_len": json_len,
        "latency_ms": resp_time_ms,
        "error": error,
    }


def run_benchmark_same_scramble(
    base_url: str,
    backends: List[str],
    scramble_min: int,
    scramble_max: int,
    trials: int,
    timeout: float,
    seed: int,
    extra_headers: Optional[Dict[str, str]] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Benchmark multiple backends on the exact same scramble instances.

    Steps:
    1) Generate a reproducible catalog of color_input instances
    2) For each entry, send the same input to all backends
    3) Return a raw results DataFrame

    Parameters
    ----------
    base_url : str
        Endpoint accepting POST with a color_input payload.
    backends : list of str
        Backend identifiers to test.
    scramble_min : int
        Minimum scramble length (inclusive).
    scramble_max : int
        Maximum scramble length (inclusive).
    trials : int
        Trials per scramble length.
    timeout : float
        Per-request timeout (seconds).
    seed : int
        Base seed for deterministic scramble generation.
    extra_headers : dict[str, str] or None, default None
        Additional request headers.
    show_progress : bool, default True
        Show progress bars.

    Returns
    -------
    pandas.DataFrame
        Raw per-request results.
    """
    catalog = make_scramble_catalog(
        scramble_min,
        scramble_max,
        trials,
        base_seed=seed,
        show_progress=show_progress,
    )

    rows: List[Dict[str, Any]] = []
    # Track per-(backend, scramble_moves) progress to compute live error rates
    progress_stats: Dict[tuple[str, int], Dict[str, int]] = {}
    total_requests = len(catalog) * len(backends)

    # Global progress bar over requests
    pbar = tqdm(
        total=total_requests,
        desc="Benchmark",
        unit="req",
        disable=not show_progress,
    )

    for backend in backends:
        # Show current backend in the progress bar description
        pbar.set_description(f"Benchmark | {backend}")

        for entry in catalog:
            scr = entry["scramble_moves"]
            trial = entry["trial"]
            color_input = entry["color_input"]
            scr_id = entry["scramble_id"]

            # Update progress bar with backend and scramble length
            pbar.set_description(f"Benchmark | {backend} | scr={scr}")

            res = benchmark_once_on_input(
                base_url=base_url,
                backend=backend,
                color_input=color_input,
                timeout=timeout,
                extra_headers=extra_headers,
            )
            row = {
                "backend": backend,
                "scramble_moves": scr,
                "trial": trial,
                "scramble_id": scr_id,
                "status_code": res["status_code"],
                "ok": res["ok"],
                "json_len": res["json_len"],
                "latency_ms": res["latency_ms"],
                "error": res["error"],
            }
            rows.append(row)

            # Update live error rate for current (backend, scr)
            key = (backend, scr)

            if key not in progress_stats:
                progress_stats[key] = {"total": 0, "success": 0}

            progress_stats[key]["total"] += 1
            progress_stats[key]["success"] += int(bool(row["ok"]))

            total = progress_stats[key]["total"]
            success = progress_stats[key]["success"]

            err_rate = 1.0 - (success / total) if total > 0 else 0.0

            # Reflect current backend, scramble and error rate in the progress bar
            pbar.set_description(
                f"Benchmark | {backend} | scr={scr} | err={err_rate:.2f}"
            )

            # With -v, avoid breaking the bar; use tqdm.write
            if logging.getLogger().level <= logging.INFO:
                tqdm.write(
                    f"{backend:14s} | scr={scr} | trial={trial:02d} | "
                    f"HTTP={row['status_code']} | ok={row['ok']} | len={row['json_len']} | "
                    f"lat={None if row['latency_ms'] is None else round(row['latency_ms'],1)} ms | "
                    f"id={scr_id}"
                )

            pbar.update(1)

    pbar.close()
    df = pd.DataFrame(rows)
    return df


def summarize_and_plot(df: pd.DataFrame, out_png: Optional[str] = None) -> None:
    """Print summary stats and plot success/latency charts.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw results from ``run_benchmark_same_scramble``.
    out_png : str or None, default None
        If provided, saves plots to ``*-success_rate.png`` and ``*-latency.png``.
    """
    df["latency_ms_num"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    # Derive error-type flags for aggregation
    err = df["error"].fillna("")
    df["is_timeout"] = err.str.startswith("Timeout(")
    df["is_json_error"] = err.str.startswith("JSONDecodeError:")
    df["is_http_error"] = err.str.startswith("HTTP ")
    df["is_req_exc"] = err.str.startswith("RequestException:")

    # Check that each scramble_id was tested by all backends
    coverage = (
        df.groupby(["scramble_id"])["backend"].nunique().rename("backends_tested")
    )
    missing = coverage[coverage < df["backend"].nunique()]
    if not missing.empty:
        logging.warning(
            "Certaines instances n'ont pas été testées par tous les backends:\n"
            + missing.to_string()
        )

    # Global summary per backend and scramble length
    agg = (
        df.groupby(["backend", "scramble_moves"])
        .agg(
            trials=("ok", "size"),
            success=("ok", "sum"),
            success_rate=("ok", "mean"),
            timeouts=("is_timeout", "sum"),
            json_errors=("is_json_error", "sum"),
            http_errors=("is_http_error", "sum"),
            request_exceptions=("is_req_exc", "sum"),
            actions_len=("json_len", "mean"),
            avg_latency_ms=("latency_ms_num", "mean"),
            med_latency_ms=("latency_ms_num", "median"),
        )
        .reset_index()
    )

    print("\n=== Résumé par backend & scramble ===")
    print(agg.to_string(index=False))

    # Graphique 1 : Success rate
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=agg,
        x="backend",
        y="success_rate",
        hue="scramble_moves",
        dodge=True,
    )
    plt.title("Taux de succès par backend (mêmes scrambles)")
    plt.ylabel("Success rate")
    plt.ylim(0, 1.05)
    plt.xlabel("")
    plt.legend(title="Scrambles")
    plt.tight_layout()

    if out_png:
        plt.savefig(
            out_png.replace(".png", "-success_rate.png"),
            dpi=150,
            bbox_inches="tight",
        )
        print(f"\nGraphiques sauvegardés : {out_png}")

    # Graphique 2 : backend latency
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=df[df["latency_ms_num"].notna()],
        x="backend",
        y="latency_ms_num",
        hue="scramble_moves",
    )
    plt.title("Latence par backend (ms) — mêmes scrambles")
    plt.ylabel("Latency (ms)")
    plt.xlabel("")
    plt.legend(title="Scrambles")
    plt.tight_layout()

    if out_png:
        plt.savefig(
            out_png.replace(".png", "-latency.png"),
            dpi=150,
            bbox_inches="tight",
        )
        print(f"\nGraphiques sauvegardés : {out_png}")

    # Graphique 3 : actions length
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=df[df["json_len"].notna()],
        x="backend",
        y="json_len",
        hue="scramble_moves",
    )
    plt.title("Nombre d'actions par backend — mêmes scrambles")
    plt.ylabel("Nombre d'actions")
    plt.xlabel("")
    plt.legend(title="Scrambles")
    plt.tight_layout()

    if out_png:
        plt.savefig(
            out_png.replace(".png", "-actions_len.png"),
            dpi=150,
            bbox_inches="tight",
        )
        print(f"\nGraphiques sauvegardés : {out_png}")

    plt.show()


def parse_headers(header_list: List[str]) -> Dict[str, str]:
    """Parse CLI ``--header"` arguments of the form "Key: Value".

    Parameters
    ----------
    header_list : list of str
        Multiple header strings.

    Returns
    -------
    dict[str, str]
        Parsed headers.

    Raises
    ------
    ValueError
        If any header does not contain a colon.
    """
    headers: Dict[str, str] = {}
    for h in header_list:
        if ":" not in h:
            raise ValueError(f'Header invalide: "{h}". Format attendu: "Key: Value"')
        k, v = h.split(":", 1)
        headers[k.strip()] = v.strip()
    return headers


def main() -> None:
    """CLI entry point for running the benchmark and plotting results."""
    parser = argparse.ArgumentParser(
        description="Benchmark API solveurs Rubik — comparaison sur EXACTEMENT les mêmes scrambles."
    )
    parser.add_argument(
        "--url", default="http://localhost:8000/", help="URL de l'endpoint POST."
    )
    parser.add_argument(
        "--backends", nargs="*", default=BACKENDS, help="Backends à tester."
    )
    parser.add_argument(
        "--scramble-min", type=int, default=1, help="Scramble minimum (inclus)."
    )
    parser.add_argument(
        "--scramble-max", type=int, default=10, help="Scramble maximum (inclus)."
    )
    parser.add_argument(
        "--trials", type=int, default=20, help="Essais par (scramble_moves)."
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="Timeout par requête (s)."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed de génération des scrambles."
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help='Headers additionnels: "Key: Value".',
    )
    parser.add_argument(
        "--save-csv",
        default="benchmark_same_scramble.csv",
        help="Chemin CSV des résultats.",
    )
    parser.add_argument(
        "--save-png",
        default="benchmark_same_scramble.png",
        help="Chemin PNG des graphes.",
    )
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Désactive les barres de progression.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity (-v, -vv).",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    extra_headers = parse_headers(args.header) if args.header else {}

    show_progress = not args.no_tqdm

    df = run_benchmark_same_scramble(
        base_url=args.url,
        backends=args.backends,
        scramble_min=args.scramble_min,
        scramble_max=args.scramble_max,
        trials=args.trials,
        timeout=args.timeout,
        seed=args.seed,
        extra_headers=extra_headers,
        show_progress=show_progress,
    )

    df.to_csv(args.save_csv, index=False)
    print(f"\nRésultats sauvegardés : {args.save_csv}")

    summarize_and_plot(df, out_png=args.save_png)


if __name__ == "__main__":
    main()
