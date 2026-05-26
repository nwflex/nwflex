"""Compound (di, tri) per-read sweep, full lflank.

Pairs are chosen so the |M|=1 bridge cannot extend either repeat:
the di motif's first base equals the tri motif's last base (R1[0]==R2[-1]).
Writes per-read shards to supplement/data_perread/compound/.
"""
import argparse
import random
from pathlib import Path

from run_batch_sweep import (
    REPO_ROOT,
    _enumerate_compound_tasks,
    _execute_tasks_parallel,
    _execute_tasks_sequential,
    _resolve_panel_path,
    _score_kwargs,
    load_panel,
)

PANEL = "data/hg38_motif_sample_K100.tsv"
OUT_DIR = "supplement/data_perread"
SEED = 20260512
N_PAIRS = 100

CMP_CFG = {
    "N_pairs": [[10, 10]],
    "bridge_lengths": [1, 2, 3, 4, 5],
    "delta1_range": [-5, 5],
    "delta2_range": [-5, 5],
    "read_len": 150,
    "k_min_flank": 1,
}


def select_pairs(panel, n_pairs, seed):
    """(di, tri) motif pairs with R1[0] == R2[-1], one seeded locus each."""
    rng = random.Random(seed + 1)
    di = panel[panel["type"].str.len() == 2]
    tri = panel[panel["type"].str.len() == 3]
    candidates = [(m1, m2)
                  for m1 in sorted(di["type"].unique())
                  for m2 in sorted(tri["type"].unique())
                  if m1[0] == m2[-1]]
    rng.shuffle(candidates)
    pairs = []
    for (m1, m2) in candidates[:n_pairs]:
        p1 = int(rng.choice(di[di["type"] == m1]["pind"].tolist()))
        p2 = int(rng.choice(tri[tri["type"] == m2]["pind"].tolist()))
        pairs.append({"motif1": m1, "motif1_len": 2, "pind1": p1,
                      "motif2": m2, "motif2_len": 3, "pind2": p2})
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    panel = load_panel(_resolve_panel_path(PANEL))
    cmp_pairs = select_pairs(panel, N_PAIRS, SEED)
    tasks = _enumerate_compound_tasks(cmp_pairs, CMP_CFG, panel)
    if args.limit is not None:
        tasks = tasks[:args.limit]
    print(f"{len(cmp_pairs)} (di,tri) pairs [R1[0]==R2[-1]] -> {len(tasks)} tasks")

    out_root = (REPO_ROOT / OUT_DIR).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    config = {"compound": CMP_CFG}
    resume = not args.no_resume
    runner = (_execute_tasks_sequential if args.workers == 0
              else _execute_tasks_parallel)
    kwargs = {} if args.workers == 0 else {"n_workers": args.workers}
    summary = runner(tasks, config, _score_kwargs(), out_root=out_root,
                     resume=resume, **kwargs)
    print(f"Done. completed={summary['completed']}, skipped={summary['skipped']}, "
          f"failed={len(summary['failed'])} (of {summary['total']}).")
    for f in summary["failed"]:
        print(f"  FAIL {f['task_id']}: {f['error']}")


if __name__ == "__main__":
    main()
