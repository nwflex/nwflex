import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "bpst", Path(__file__).resolve().parent.parent / "scripts" / "build_perf_summary_table.py"
)
bpst = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bpst)


def test_both_flanks_positive_drops_zero_flank_reads():
    df = pd.DataFrame({
        "lflank": [5, 0, 3], "rflank": [4, 2, 0],
        "fwd_state": ["P", "P", "P"], "rc_state": ["P", "P", "P"],
    })
    kept = bpst._both_flanks_positive(df)
    assert len(kept) == 1
    assert kept.iloc[0]["lflank"] == 5


def test_pooled_strand_correct_counts_each_strand_independently():
    # one group "g"=1: fwd P + rc M -> 2 obs, 1 correct -> 0.5
    df = pd.DataFrame({"g": [1], "fwd_state": ["P"], "rc_state": ["M"]})
    out = bpst._pooled_strand_correct(df, ["g"])
    row = out.iloc[0]
    assert row["n_obs"] == 2
    assert row["n_correct"] == 1
    assert row["prop_correct"] == 0.5


def test_pooled_strand_correct_T_counts_as_incorrect():
    # length-only metric: T (co-optimal tie) is NOT correct
    df = pd.DataFrame({"g": [1, 1], "fwd_state": ["T", "P"], "rc_state": ["T", "D"]})
    out = bpst._pooled_strand_correct(df, ["g"])
    row = out.iloc[0]
    assert row["n_obs"] == 4         # 2 reads x 2 strands
    assert row["n_correct"] == 1     # only the single P


def test_single_table_groups_and_labels():
    df = pd.DataFrame({
        "motif_len": [3, 3], "snv_offset": [-1, -1], "delta": [1, 1],
        "arm": ["NW-flex", "NW-flex"], "lflank": [5, 5], "rflank": [5, 5],
        "fwd_state": ["P", "P"], "rc_state": ["P", "M"],
    })
    out = bpst.single_table(df)
    row = out.iloc[0]
    assert row["test"] == "single"
    assert row["aligner"] == "NW-flex"
    assert row["n_obs"] == 4 and row["n_correct"] == 3
    assert "delta" in out.columns and row["delta"] == 1


def test_compound_table_excludes_zero_zero_cell():
    df = pd.DataFrame({
        "motif1_len": [2, 2], "motif2_len": [3, 3], "bridge_len": [1, 1],
        "delta1": [0, 1], "delta2": [0, -1], "arm": ["NW-flex", "NW-flex"],
        "lflank": [5, 5], "rflank": [5, 5],
        "fwd_state": ["M", "P"], "rc_state": ["M", "P"],
    })
    out = bpst.compound_table(df)
    row = out.iloc[0]
    assert row["test"] == "compound"
    # the (0,0) row is dropped, leaving the (1,-1) row: 2 obs, both P
    assert row["n_obs"] == 2 and row["n_correct"] == 2
    assert row["bridge_len"] == 1


def test_to_output_schema_has_columns_and_nulls():
    single_g = pd.DataFrame({
        "motif_len": [3], "snv_offset": [-1], "delta": [1], "aligner": ["NW-flex"],
        "n_correct": [4], "n_obs": [4], "prop_correct": [1.0], "test": ["single"],
    })
    compound_g = pd.DataFrame({
        "motif1_len": [2], "motif2_len": [3], "bridge_len": [1], "aligner": ["NW-flex"],
        "n_correct": [2], "n_obs": [2], "prop_correct": [1.0], "test": ["compound"],
    })
    out = bpst._to_output_schema(single_g, compound_g)
    assert list(out.columns) == bpst.OUTPUT_COLUMNS
    s = out[out["test"] == "single"].iloc[0]
    c = out[out["test"] == "compound"].iloc[0]
    assert pd.isna(s["motif1_len"]) and pd.isna(s["bridge_len"])
    assert pd.isna(c["motif_len"]) and pd.isna(c["delta"])
    assert s["delta"] == 1 and c["bridge_len"] == 1


def test_filter_paper_restricts_single_and_compound():
    single_df = pd.DataFrame({
        "motif_len": [3, 2, 3], "N": [10, 10, 5], "snv_offset": [-1, -1, -1],
    })
    compound_df = pd.DataFrame({
        "motif1_len": [2, 1], "motif2_len": [3, 2],
    })
    s, c = bpst.filter_paper(single_df, compound_df)
    # only the tri/N=10 single row survives
    assert len(s) == 1 and s.iloc[0]["motif_len"] == 3 and s.iloc[0]["N"] == 10
    # only the (di, tri) compound row survives
    assert len(c) == 1 and c.iloc[0]["motif1_len"] == 2 and c.iloc[0]["motif2_len"] == 3


def _write_single_shard(path):
    pd.DataFrame({
        "delta": [1, 1], "lflank": [5, 5], "rflank": [5, 5],
        "arm": ["NW-flex", "BWA-std"], "fwd_state": ["P", "M"], "rc_state": ["P", "M"],
        "state": ["P", "M"], "motif_len": [3, 3], "N": [10, 10], "snv_offset": [-1, -1],
    }).to_csv(path, index=False)


def _write_compound_shard(path):
    pd.DataFrame({
        "delta1": [1, 0], "delta2": [-1, 0], "lflank": [5, 5], "rflank": [5, 5],
        "arm": ["NW-flex", "NW-flex"], "fwd_state": ["P", "M"], "rc_state": ["P", "M"],
        "state": ["P", "M"], "motif1_len": [2, 2], "motif2_len": [3, 3],
        "bridge_len": [1, 1],
    }).to_csv(path, index=False)


def test_build_table_paper_end_to_end(tmp_path):
    sd = tmp_path / "single"; sd.mkdir(); _write_single_shard(sd / "s01.csv")
    cd = tmp_path / "compound"; cd.mkdir(); _write_compound_shard(cd / "c01.csv")
    out = bpst.build_table("paper", single_dir=sd, compound_dir=cd)
    assert list(out.columns) == bpst.OUTPUT_COLUMNS
    nwflex_single = out[(out["test"] == "single") & (out["aligner"] == "NW-flex")].iloc[0]
    assert nwflex_single["prop_correct"] == 1.0   # both strands P
    # compound (0,0) row excluded -> only the (1,-1) NW-flex row, both strands P
    nwflex_comp = out[(out["test"] == "compound") & (out["aligner"] == "NW-flex")].iloc[0]
    assert nwflex_comp["n_obs"] == 2 and nwflex_comp["prop_correct"] == 1.0


def test_check_paper_flags_mismatch_and_passes_match():
    # build a compound frame matching NW-flex=1.0 but BWA-std wrong
    rows = []
    for m in [1, 2, 3, 4, 5]:
        rows.append({"test": "compound", "aligner": "NW-flex", "bridge_len": m,
                     "prop_correct": 1.0})
        rows.append({"test": "compound", "aligner": "BWA-std", "bridge_len": m,
                     "prop_correct": 0.0})  # deliberately wrong
        rows.append({"test": "compound", "aligner": "BWA-no-clip", "bridge_len": m,
                     "prop_correct": bpst.PAPER_TABLE_1B["BWA-no-clip"][m - 1]})
    out = pd.DataFrame(rows)
    problems = bpst.check_paper(out)
    assert any("BWA-std" in p for p in problems)
    assert not any("NW-flex" in p for p in problems)


def test_main_writes_csv(tmp_path, monkeypatch):
    sd = tmp_path / "single"; sd.mkdir(); _write_single_shard(sd / "s01.csv")
    cd = tmp_path / "compound"; cd.mkdir(); _write_compound_shard(cd / "c01.csv")
    out_csv = tmp_path / "perf_summary.csv"
    argv = ["prog", "--mode", "paper", "--single-dir", str(sd),
            "--compound-dir", str(cd), "--out", str(out_csv)]
    monkeypatch.setattr("sys.argv", argv)
    bpst.main()
    assert out_csv.exists()
    loaded = pd.read_csv(out_csv)
    assert list(loaded.columns) == bpst.OUTPUT_COLUMNS
    assert (loaded["prop_correct"].dropna().between(0, 1)).all()
