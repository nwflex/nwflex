"""Coordinator for the comprehensive batch sweep.

Single-locus tasks are dispatched via :class:`ProcessPoolExecutor`.  Each
task runs the same simulation flow the notebooks use
(``nwflex.simulation``) and writes one tidy-frame parquet shard to
``output.data_dir``.  Cross-locus aggregation, figures, and tables are
produced by sibling scripts.

Usage::

    python scripts/run_batch_sweep.py --config scripts/configs/single_repeat.yaml [--dry-run]
    python scripts/run_batch_sweep.py --config scripts/configs/compound.yaml [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
import yaml
from tqdm.auto import tqdm

from nwflex.default import get_default_scoring
from nwflex.ep_patterns import build_EP_STR_phase
from nwflex.repeats import STRLocus
from nwflex.simulation import (
    BWACompoundMethod,
    BWAMethod,
    CompoundLocus,
    NWFlexCompoundMethod,
    NWFlexMethod,
    build_compound_haplotype,
    build_compound_mirror_frame,
    build_haplotype,
    build_locus_from_panel,
    build_mirror_frame,
    pivot_for_heatmap,
    sweep,
    wrap_methods_for_multizone_truth,
)
from nwflex.simulation.core import clean_flank_window
from nwflex.simulation.sweep import make_variant
from nwflex.ep_patterns import build_EP_multi_STR_phase


REPO_ROOT = Path(__file__).resolve().parent.parent


def _score_kwargs(name: str = "bwa_mem") -> dict:
    score_matrix, gap_open, gap_extend, a2i = get_default_scoring(name)
    return dict(score_matrix=score_matrix, gap_open=gap_open,
                gap_extend=gap_extend, alphabet_to_index=a2i)


def _next_base(b: str) -> str:
    """Return a single base distinct from ``b`` (deterministic A→C→G→T→A)."""
    return {"A": "C", "C": "G", "G": "T", "T": "A"}[b.upper()]


def run_single_repeat_task(
    *,
    pind: int,
    motif: str,
    panel_lflank: str,
    panel_rflank: str,
    N: int,
    snv_offset: Optional[int],
    delta_min: int,
    delta_max: int,
    read_len: int,
    k_min_flank: int,
    target_lflanks: Optional[Sequence[int]],
    score_kwargs: Mapping[str, Any],
    flank_len: int = 200,
    nwflex_factor: int = 3,
) -> pd.DataFrame:
    """Run one ``(locus, N, snv_offset)`` cell of the single-repeat sweep.

    Builds the locus, the NW-flex extended reference, the mirror frame,
    and the EP patterns; runs the sweep harness across ``delta_min..
    delta_max`` (with optional flank SNV); returns a tidy DataFrame
    with one row per ``(delta, lflank, arm)`` cell, carrying the
    per-strand and combined verdicts plus locus / parameter metadata.

    ``snv_offset = k`` places the SNV at the ``k``-th base inside the
    left flank counting from the boundary (``snv_offset=0`` is the
    flank base immediately adjacent to the repeat).
    """
    locus = build_locus_from_panel(
        motif=motif,
        panel_lflank=panel_lflank,
        panel_rflank=panel_rflank,
        ref_n=N,
        flank_len=flank_len,
    )
    nwflex_locus = STRLocus(
        A=locus.A, R=locus.R, N=nwflex_factor * locus.N, B=locus.B,
    )
    # Mirror frame: we don't have reads yet, so pass [] and use the
    # returned zone / reference fields only.
    (rc_locus_X, _, rc_zone,
     rc_nwflex_X, rc_nwflex_zone) = build_mirror_frame(
        reference=locus.X, reads=[], zone=(locus.s, locus.e),
        extra_reference=nwflex_locus.X,
        extra_zone=(nwflex_locus.s, nwflex_locus.e),
    )
    ep_fwd = build_EP_STR_phase(
        nwflex_locus.n, nwflex_locus.s, nwflex_locus.e, nwflex_locus.k,
    )
    ep_rc = build_EP_STR_phase(
        nwflex_locus.n, *rc_nwflex_zone, nwflex_locus.k,
    )

    methods = [
        BWAMethod("BWA-std", locus, rc_locus_X, rc_zone,
                  no_clip=False, score_kwargs=score_kwargs),
        BWAMethod("BWA-no-clip", locus, rc_locus_X, rc_zone,
                  no_clip=True, score_kwargs=score_kwargs),
        NWFlexMethod("NW-flex", nwflex_locus, rc_nwflex_X, rc_nwflex_zone,
                     ep_fwd, ep_rc, score_kwargs=score_kwargs),
    ]

    # Locate the SNV in absolute coordinates (left flank, anchored at the
    # boundary) and pick a substituted base deterministically.
    snv_abs_pos: Optional[int] = None
    snv_base_new: Optional[str] = None
    snv_base_orig: Optional[str] = None
    if snv_offset is not None:
        snv_abs_pos = locus.s - 1 - snv_offset
        if snv_abs_pos < 0:
            raise ValueError(
                f"snv_offset={snv_offset} would land before the start of "
                f"the haplotype (flank_len={flank_len}, locus.s={locus.s})"
            )

    variants = []
    target_set = None if target_lflanks is None else set(target_lflanks)
    for delta in range(delta_min, delta_max + 1):
        if locus.N + delta < 0:
            continue
        snv: Optional[tuple[int, str]] = None
        if snv_abs_pos is not None:
            # Build a temporary unmutated haplotype to look up the
            # original base; this is cheap.
            tmp = build_haplotype(locus, delta=delta)
            snv_base_orig = tmp.sequence[snv_abs_pos]
            snv_base_new = _next_base(snv_base_orig)
            snv = (snv_abs_pos, snv_base_new)
        hap = build_haplotype(locus, delta=delta, snv=snv)
        v = make_variant(
            label={"delta": delta},
            hap=hap,
            read_len=read_len,
            k_min_flank=k_min_flank,
            target_lflanks=target_set,
        )
        if v.reads:
            variants.append(v)

    if not variants:
        return pd.DataFrame()

    long_df = sweep(variants, methods)
    grid_df = pivot_for_heatmap(long_df, combine="best")

    grid_df["pind"] = pind
    grid_df["motif"] = motif
    grid_df["motif_len"] = len(motif)
    grid_df["N"] = N
    grid_df["snv_offset"] = (-1 if snv_offset is None else snv_offset)
    grid_df["snv_base_orig"] = snv_base_orig
    grid_df["snv_base_new"] = snv_base_new
    grid_df["sweep_kind"] = "single_repeat"
    return grid_df


def _split_bridge_length(bridge_len: int) -> tuple[int, int]:
    """Deterministic split of a target bridge length into
    ``(bridge_n1, bridge_n2)``.  Convention: bridge_n1 >= bridge_n2,
    so the left side carries the extra base when the total is odd.
    """
    bridge_n1 = (bridge_len + 1) // 2
    bridge_n2 = bridge_len - bridge_n1
    return bridge_n1, bridge_n2


def run_compound_task(
    *,
    pind1: int, motif1: str,
    panel_lflank_1: str, panel_rflank_1: str,
    pind2: int, motif2: str,
    panel_lflank_2: str, panel_rflank_2: str,
    N1: int, N2: int,
    bridge_len: int,
    delta1_min: int, delta1_max: int,
    delta2_min: int, delta2_max: int,
    read_len: int, k_min_flank: int,
    score_kwargs: Mapping[str, Any],
    flank_len: int = 200,
    nwflex_factor: int = 3,
) -> pd.DataFrame:
    """Run one ``(motif-pair, N-pair, bridge-len)`` compound task.

    Builds the compound locus from explicit motif rows (independent of
    ``build_compound_locus_from_panel`` so we can pick exact motifs
    rather than ``.iloc[0]`` of a length filter); sweeps
    ``(Δ1, Δ2)``; returns a tidy DataFrame with one row per
    ``(delta1, delta2, lflank, arm)`` read cell, retaining the full
    lflank range and separate ``fwd_state`` / ``rc_state``.
    """
    bridge_n1, bridge_n2 = _split_bridge_length(bridge_len)
    A = clean_flank_window(motif1, panel_lflank_1, flank_len, "left")
    B = clean_flank_window(motif2, panel_rflank_2, flank_len, "right")
    M_left = (clean_flank_window(motif1, panel_rflank_1, bridge_n1, "right")
              if bridge_n1 > 0 else "")
    M_right = (clean_flank_window(motif2, panel_lflank_2, bridge_n2, "left")
               if bridge_n2 > 0 else "")
    compound = CompoundLocus(
        A=A, R1=motif1, N1=N1, M=M_left + M_right,
        R2=motif2, N2=N2, B=B, nwflex_factor=nwflex_factor,
    )

    (rc_X, rc_X_ext, _, rc_zones, rc_zones_ext) = build_compound_mirror_frame(
        compound, reads=[],
    )
    ep_fwd = build_EP_multi_STR_phase(
        n=len(compound.X_ext),
        blocks=[(compound.zones_ext[0][0], compound.zones_ext[0][1], compound.k1),
                (compound.zones_ext[1][0], compound.zones_ext[1][1], compound.k2)],
    )
    ep_rc = build_EP_multi_STR_phase(
        n=len(rc_X_ext),
        blocks=[(rc_zones_ext[0][0], rc_zones_ext[0][1], compound.k1),
                (rc_zones_ext[1][0], rc_zones_ext[1][1], compound.k2)],
    )

    methods = [
        BWACompoundMethod("BWA-std", compound, rc_X, rc_zones,
                          no_clip=False, score_kwargs=score_kwargs),
        BWACompoundMethod("BWA-no-clip", compound, rc_X, rc_zones,
                          no_clip=True, score_kwargs=score_kwargs),
        NWFlexCompoundMethod("NW-flex", compound, rc_X_ext, rc_zones_ext,
                             ep_fwd, ep_rc, score_kwargs=score_kwargs),
    ]

    variants = []
    for d1 in range(delta1_min, delta1_max + 1):
        if N1 + d1 < 0:
            continue
        for d2 in range(delta2_min, delta2_max + 1):
            if N2 + d2 < 0:
                continue
            hap = build_compound_haplotype(compound, delta1=d1, delta2=d2)
            v = make_variant(
                label={"delta1": d1, "delta2": d2},
                hap=hap,
                read_len=read_len, k_min_flank=k_min_flank,
            )
            if v.reads:
                variants.append(v)

    if not variants:
        return pd.DataFrame()

    long_df = sweep(
        variants,
        wrap_methods_for_multizone_truth(methods, variants),
    )
    # Retain per-read rows (one per (delta1, delta2, lflank, arm), full
    # lflank range) so downstream can pool reads under the length-only
    # correctness metric. fwd_state / rc_state stay separate.
    out_df = pivot_for_heatmap(long_df, combine="best")

    out_df["pind1"] = pind1
    out_df["pind2"] = pind2
    out_df["motif1"] = motif1
    out_df["motif2"] = motif2
    out_df["motif1_len"] = len(motif1)
    out_df["motif2_len"] = len(motif2)
    out_df["N1"] = N1
    out_df["N2"] = N2
    out_df["bridge_len"] = bridge_len
    out_df["bridge_n1"] = bridge_n1
    out_df["bridge_n2"] = bridge_n2
    out_df["sweep_kind"] = "compound"
    return out_df


def load_config(path: Path) -> Mapping[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def load_panel(panel_path: Path) -> pd.DataFrame:
    """Load the panel TSV and drop loci with ambiguous bases (``N``) in
    either flank — those break BWA's score matrix lookup and the
    simulation pipeline assumes a clean ACGT alphabet."""
    panel = pd.read_csv(panel_path, sep="\t")
    panel["motif_len"] = panel["type"].str.len()
    clean = ~(
        panel["lflank"].str.contains("[^ACGT]", regex=True)
        | panel["rflank"].str.contains("[^ACGT]", regex=True)
    )
    dropped = (~clean).sum()
    if dropped:
        print(f"  (filtered {dropped} panel row(s) with non-ACGT flank bases)")
    return panel[clean].reset_index(drop=True)


def select_single_repeat_loci(
    panel: pd.DataFrame,
    motif_lengths: Sequence[int],
    loci_per_length: int,
    seed: int,
) -> list[dict]:
    """Sample ``loci_per_length`` panel rows per motif length, round-robin
    over distinct motif sequences so every motif gets representation
    before any motif gets a second locus.
    """
    rng = random.Random(seed)
    selected: list[dict] = []
    for L in motif_lengths:
        sub = panel[panel["motif_len"] == L]
        motifs = sorted(sub["type"].unique())
        per_motif_pool = {
            m: sub[sub["type"] == m]["pind"].tolist() for m in motifs
        }
        for pool in per_motif_pool.values():
            rng.shuffle(pool)
        picked: list[dict] = []
        while len(picked) < loci_per_length and any(per_motif_pool.values()):
            for m in motifs:
                if not per_motif_pool[m]:
                    continue
                picked.append({
                    "pind": int(per_motif_pool[m].pop(0)),
                    "motif": m,
                    "motif_len": L,
                })
                if len(picked) >= loci_per_length:
                    break
        selected.extend(picked)
    return selected


def select_compound_motif_pairs(
    panel: pd.DataFrame,
    motif_length_pairs: Sequence[Sequence[int]],
    pairs_per_length_pair: int,
    seed: int,
) -> list[dict]:
    """For each motif-length pair ``(L1, L2)``, sample
    ``pairs_per_length_pair`` ``(motif1, motif2)`` combinations and pick
    one panel locus per motif from a seeded shuffle.
    """
    rng = random.Random(seed + 1)
    selected: list[dict] = []
    for (L1, L2) in motif_length_pairs:
        sub1 = panel[panel["motif_len"] == L1]
        sub2 = panel[panel["motif_len"] == L2]
        motifs1 = sorted(sub1["type"].unique())
        motifs2 = sorted(sub2["type"].unique())
        candidates = [(m1, m2) for m1 in motifs1 for m2 in motifs2
                      if not (L1 == L2 and m1 == m2)]
        rng.shuffle(candidates)
        for (m1, m2) in candidates[:pairs_per_length_pair]:
            pind1 = int(rng.choice(
                sub1[sub1["type"] == m1]["pind"].tolist()
            ))
            pind2 = int(rng.choice(
                sub2[sub2["type"] == m2]["pind"].tolist()
            ))
            selected.append({
                "motif1": m1, "motif1_len": L1, "pind1": pind1,
                "motif2": m2, "motif2_len": L2, "pind2": pind2,
            })
    return selected


def _resolve_panel_path(config_panel: str) -> Path:
    p = Path(config_panel)
    if p.is_absolute():
        return p
    return (REPO_ROOT / config_panel).resolve()


def _print_selection(sr_loci: list[dict], cmp_pairs: list[dict]) -> None:
    print(f"\nSelected {len(sr_loci)} single-repeat loci:")
    if sr_loci:
        sr_df = pd.DataFrame(sr_loci)
        for L, g in sr_df.groupby("motif_len"):
            motifs = g["motif"].tolist()
            pinds = g["pind"].tolist()
            print(f"  L={L} ({len(g)} loci): motifs={motifs} pinds={pinds}")

    print(f"\nSelected {len(cmp_pairs)} compound motif pairs:")
    if cmp_pairs:
        cmp_df = pd.DataFrame(cmp_pairs)
        for (l1, l2), g in cmp_df.groupby(["motif1_len", "motif2_len"]):
            pairs = [(r.motif1, r.motif2) for r in g.itertuples()]
            print(f"  ({l1},{l2}): {pairs}")


def _enumerate_single_repeat_tasks(
    sr_loci: list[dict],
    sr_cfg: Mapping[str, Any],
    panel: pd.DataFrame,
) -> list[dict]:
    """Cartesian product of selected loci × N_values × snv_positions.

    Each task carries everything the worker needs to call
    :func:`run_single_repeat_task` — including the panel row's flank
    sequences — so workers don't need to ship the panel across the
    process boundary.
    """
    panel_by_pind = panel.set_index("pind")
    tasks: list[dict] = []
    for locus in sr_loci:
        row = panel_by_pind.loc[locus["pind"]]
        for N in sr_cfg["N_values"]:
            for snv_pos in sr_cfg["snv_positions"]:
                offset = snv_pos  # may be None or int
                task_id = (f"pind{locus['pind']:05d}"
                           f"__N{N:02d}"
                           f"__SNV{(-1 if offset is None else offset):+d}")
                tasks.append({
                    "kind": "single_repeat",
                    "task_id": task_id,
                    "pind": locus["pind"],
                    "motif": locus["motif"],
                    "N": N,
                    "snv_offset": offset,
                    "panel_lflank": row["lflank"],
                    "panel_rflank": row["rflank"],
                })
    return tasks


def _enumerate_compound_tasks(
    cmp_pairs: list[dict],
    cmp_cfg: Mapping[str, Any],
    panel: pd.DataFrame,
) -> list[dict]:
    """Cartesian product of selected motif pairs × N_pairs × bridge_lengths.

    Each task carries both motif rows' flank strings so workers don't
    need the panel.
    """
    panel_by_pind = panel.set_index("pind")
    tasks: list[dict] = []
    for pair in cmp_pairs:
        row1 = panel_by_pind.loc[pair["pind1"]]
        row2 = panel_by_pind.loc[pair["pind2"]]
        for (N1, N2) in cmp_cfg["N_pairs"]:
            for bridge_len in cmp_cfg["bridge_lengths"]:
                task_id = (
                    f"pair{pair['pind1']:05d}_{pair['pind2']:05d}"
                    f"__N{N1:02d}_{N2:02d}"
                    f"__M{bridge_len}"
                )
                tasks.append({
                    "kind": "compound",
                    "task_id": task_id,
                    "pind1": pair["pind1"], "motif1": pair["motif1"],
                    "pind2": pair["pind2"], "motif2": pair["motif2"],
                    "panel_lflank_1": row1["lflank"],
                    "panel_rflank_1": row1["rflank"],
                    "panel_lflank_2": row2["lflank"],
                    "panel_rflank_2": row2["rflank"],
                    "N1": N1, "N2": N2,
                    "bridge_len": bridge_len,
                })
    return tasks


def _shard_path(out_root: Path, task: Mapping[str, Any]) -> Path:
    return out_root / task["kind"] / f"{task['task_id']}.csv"


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.rename(path)


def _run_one_task(
    task: Mapping[str, Any],
    config: Mapping[str, Any],
    score_kw: dict,
) -> pd.DataFrame:
    """Dispatch one enumerated task to the appropriate task function.

    Looks up the kind-specific sub-config block from ``config``.
    """
    sweep_cfg = config[task["kind"]]
    if task["kind"] == "single_repeat":
        # ``target_lflanks: null`` (or missing) means "sweep the full
        # informative lflank range" — reads carry their own lflank/rflank
        # extents so downstream code can filter to both-flanks-positive.
        tl = sweep_cfg.get("target_lflanks", list(range(1, 11)))
        return run_single_repeat_task(
            pind=int(task["pind"]),
            motif=task["motif"],
            panel_lflank=task["panel_lflank"],
            panel_rflank=task["panel_rflank"],
            N=task["N"],
            snv_offset=task["snv_offset"],
            delta_min=sweep_cfg["delta_range"][0],
            delta_max=sweep_cfg["delta_range"][1],
            read_len=sweep_cfg["read_len"],
            k_min_flank=sweep_cfg["k_min_flank"],
            target_lflanks=tl,
            score_kwargs=score_kw,
        )
    if task["kind"] == "compound":
        return run_compound_task(
            pind1=int(task["pind1"]), motif1=task["motif1"],
            panel_lflank_1=task["panel_lflank_1"],
            panel_rflank_1=task["panel_rflank_1"],
            pind2=int(task["pind2"]), motif2=task["motif2"],
            panel_lflank_2=task["panel_lflank_2"],
            panel_rflank_2=task["panel_rflank_2"],
            N1=task["N1"], N2=task["N2"],
            bridge_len=task["bridge_len"],
            delta1_min=sweep_cfg["delta1_range"][0],
            delta1_max=sweep_cfg["delta1_range"][1],
            delta2_min=sweep_cfg["delta2_range"][0],
            delta2_max=sweep_cfg["delta2_range"][1],
            read_len=sweep_cfg["read_len"],
            k_min_flank=sweep_cfg["k_min_flank"],
            score_kwargs=score_kw,
        )
    raise ValueError(f"unknown task kind: {task['kind']!r}")


def _worker(task: Mapping[str, Any],
            config: Mapping[str, Any],
            score_kw: dict,
            out_root: Path) -> dict:
    """Run one task and write its shard.  Module-level so it pickles
    cleanly for :class:`ProcessPoolExecutor`."""
    try:
        df = _run_one_task(task, config, score_kw)
        shard = _shard_path(out_root, task)
        _atomic_write_csv(df, shard)
        return {"task_id": task["task_id"], "ok": True, "n_rows": len(df)}
    except Exception as e:
        return {"task_id": task["task_id"], "ok": False, "error": repr(e)}


def _smoke_test_compound(panel: pd.DataFrame, score_kw: dict) -> None:
    # Use the same panel rows the cleaned NB9 uses to spot-check.
    row1 = panel[panel["type"].str.len() == 2].iloc[0]
    row2 = panel[panel["type"].str.len() == 3].iloc[0]
    for bridge_len in [2, 5]:
        df = run_compound_task(
            pind1=int(row1["pind"]), motif1=row1["type"],
            panel_lflank_1=row1["lflank"], panel_rflank_1=row1["rflank"],
            pind2=int(row2["pind"]), motif2=row2["type"],
            panel_lflank_2=row2["lflank"], panel_rflank_2=row2["rflank"],
            N1=10, N2=10,
            bridge_len=bridge_len,
            delta1_min=-5, delta1_max=5,
            delta2_min=-5, delta2_max=5,
            read_len=150, k_min_flank=1,
            score_kwargs=score_kw,
        )
        bridge_n1, bridge_n2 = _split_bridge_length(bridge_len)
        print(f"\nCompound smoke (|M|={bridge_len}={bridge_n1}+{bridge_n2}, "
              f"motifs={row1['type']!r}/{row2['type']!r}):")
        print(f"  shape: {df.shape}")
        print(df.groupby("arm")["state"]
                .value_counts().unstack(fill_value=0)
                .reindex(columns=["P", "T", "M", "D"], fill_value=0))


def _smoke_test_single_repeat(panel: pd.DataFrame, score_kw: dict) -> None:
    row = panel[panel["type"].str.len() == 3].iloc[0]
    df = run_single_repeat_task(
        pind=int(row["pind"]),
        motif=row["type"],
        panel_lflank=row["lflank"],
        panel_rflank=row["rflank"],
        N=10,
        snv_offset=None,
        delta_min=-5, delta_max=5,
        read_len=150, k_min_flank=1,
        target_lflanks=list(range(1, 11)),
        score_kwargs=score_kw,
    )
    print(f"\nSmoke test (no SNV): pind={row['pind']} motif={row['type']!r} N=10")
    print(f"  shape: {df.shape}, "
          f"arms: {df['arm'].unique().tolist()}, "
          f"deltas: {sorted(df['delta'].unique())}, "
          f"lflanks: {sorted(df['lflank'].unique())}")
    print(df.groupby("arm")["state"]
            .value_counts().unstack(fill_value=0)
            .reindex(columns=["P", "T", "M", "D"], fill_value=0))

    df_snv = run_single_repeat_task(
        pind=int(row["pind"]),
        motif=row["type"],
        panel_lflank=row["lflank"],
        panel_rflank=row["rflank"],
        N=10,
        snv_offset=2,
        delta_min=-5, delta_max=5,
        read_len=150, k_min_flank=1,
        target_lflanks=list(range(1, 11)),
        score_kwargs=score_kw,
    )
    print(f"\nSmoke test (SNV at offset 2): pind={row['pind']} motif={row['type']!r} N=10")
    print(f"  SNV: {df_snv['snv_base_orig'].iloc[0]} -> "
          f"{df_snv['snv_base_new'].iloc[0]}")
    print(df_snv.groupby("arm")["state"]
            .value_counts().unstack(fill_value=0)
            .reindex(columns=["P", "T", "M", "D"], fill_value=0))


def _filter_pending(
    tasks: list[dict], out_root: Path, resume: bool,
) -> tuple[list[dict], int]:
    """Split tasks into (pending, skipped_count) based on shard existence."""
    if not resume:
        return list(tasks), 0
    pending: list[dict] = []
    skipped = 0
    for task in tasks:
        if _shard_path(out_root, task).exists():
            skipped += 1
        else:
            pending.append(task)
    return pending, skipped


def _execute_tasks_sequential(
    tasks: list[dict],
    config: Mapping[str, Any],
    score_kw: dict,
    out_root: Path,
    resume: bool,
) -> dict:
    """Run tasks one at a time, writing one CSV shard per task.

    Returns a summary dict (counts of completed / skipped / failed).
    """
    pending, skipped = _filter_pending(tasks, out_root, resume)
    completed = 0
    failed: list[dict] = []
    with tqdm(total=len(pending), desc="sweep", unit="task") as pbar:
        for task in pending:
            result = _worker(task, config, score_kw, out_root)
            if result["ok"]:
                completed += 1
            else:
                failed.append(result)
                pbar.write(f"FAIL {result['task_id']}: {result['error']}")
            pbar.update(1)
    return {"completed": completed, "skipped": skipped,
            "failed": failed, "total": len(tasks)}


def _execute_tasks_parallel(
    tasks: list[dict],
    config: Mapping[str, Any],
    score_kw: dict,
    out_root: Path,
    resume: bool,
    n_workers: int,
) -> dict:
    """Run tasks via :class:`ProcessPoolExecutor` with a tqdm progress
    bar.  Each worker writes its own shard atomically; shard existence
    is the resume signal.
    """
    pending, skipped = _filter_pending(tasks, out_root, resume)
    completed = 0
    failed: list[dict] = []
    if not pending:
        return {"completed": 0, "skipped": skipped,
                "failed": [], "total": len(tasks)}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(_worker, task, config, score_kw, out_root)
            for task in pending
        ]
        with tqdm(total=len(futures), desc=f"sweep ({n_workers}w)",
                  unit="task") as pbar:
            for fut in as_completed(futures):
                result = fut.result()
                if result["ok"]:
                    completed += 1
                else:
                    failed.append(result)
                    pbar.write(f"FAIL {result['task_id']}: "
                               f"{result['error']}")
                pbar.update(1)
    return {"completed": completed, "skipped": skipped,
            "failed": failed, "total": len(tasks)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to the sweep config YAML.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Select loci + motif pairs and print; do not "
                             "execute the sweep.")
    parser.add_argument("--smoke-test-single", action="store_true",
                        help="Run one single-repeat task on the first "
                             "panel trinucleotide and print the state counts.")
    parser.add_argument("--smoke-test-compound", action="store_true",
                        help="Run two compound tasks (|M|=2 and |M|=5) "
                             "on the first panel di/trinucleotides.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N enumerated tasks "
                             "(for testing).")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-run tasks even if their shard already "
                             "exists.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker process count for ProcessPoolExecutor. "
                             "0 = sequential (in-process). Default: 0 if "
                             "--limit is small (<= 4), else min(cpu_count, 8).")
    args = parser.parse_args()

    config = load_config(args.config)
    panel_path = _resolve_panel_path(config["panel"])
    panel = load_panel(panel_path)
    print(f"Loaded panel: {len(panel)} rows from {panel_path}")

    sr_cfg = config["single_repeat"]
    sr_loci = select_single_repeat_loci(
        panel,
        motif_lengths=sr_cfg["motif_lengths"],
        loci_per_length=sr_cfg["loci_per_length"],
        seed=config["seed"],
    )

    cmp_cfg = config["compound"]
    cmp_pairs = select_compound_motif_pairs(
        panel,
        motif_length_pairs=cmp_cfg["motif_length_pairs"],
        pairs_per_length_pair=cmp_cfg["pairs_per_length_pair"],
        seed=config["seed"],
    )

    _print_selection(sr_loci, cmp_pairs)

    if args.smoke_test_single:
        _smoke_test_single_repeat(panel, _score_kwargs())
        return

    if args.smoke_test_compound:
        _smoke_test_compound(panel, _score_kwargs())
        return

    if args.dry_run:
        return

    sr_tasks = _enumerate_single_repeat_tasks(sr_loci, sr_cfg, panel)
    cmp_tasks = _enumerate_compound_tasks(cmp_pairs, cmp_cfg, panel)
    all_tasks = sr_tasks + cmp_tasks
    print(f"\nEnumerated {len(sr_tasks)} single-repeat + "
          f"{len(cmp_tasks)} compound = {len(all_tasks)} tasks total.")
    if args.limit is not None:
        all_tasks = all_tasks[:args.limit]
        print(f"  --limit applied: running first {len(all_tasks)}.")

    out_root = (REPO_ROOT / config["output"]["data_dir"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {out_root}")

    if args.workers is None:
        n_workers = (0 if len(all_tasks) <= 4
                     else min(os.cpu_count() or 1, 8))
    else:
        n_workers = args.workers

    score_kw = _score_kwargs()
    resume = not args.no_resume
    if n_workers == 0:
        summary = _execute_tasks_sequential(
            all_tasks, config, score_kw, out_root=out_root, resume=resume,
        )
    else:
        summary = _execute_tasks_parallel(
            all_tasks, config, score_kw, out_root=out_root, resume=resume,
            n_workers=n_workers,
        )
    print(f"\nDone. completed={summary['completed']}, "
          f"skipped={summary['skipped']}, "
          f"failed={len(summary['failed'])} (of {summary['total']}).")
    if summary["failed"]:
        for f in summary["failed"]:
            print(f"  FAIL {f['task_id']}: {f['error']}")


if __name__ == "__main__":
    main()
