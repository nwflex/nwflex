# test_trf.py
#
# Tests for nwflex.trf — TRF parsing, isolation annotation, and filtering.
# Uses the chr21 snippet fixture in data/chr21_snippet_demo.dat.

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nwflex.trf import (
    annotate_repeat_isolation,
    filter_isolated_repeats,
    parse_trf_dat,
    read_chrom_lengths,
)


REPO_DATA = Path(__file__).resolve().parent.parent / "data"
TRF_DAT = REPO_DATA / "chr21_snippet_demo.dat"


# ============================================================================
# PARSE_TRF_DAT
# ============================================================================

class TestParseTrfDat:
    def test_basic_shape(self):
        df = parse_trf_dat(TRF_DAT)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        # 15 TRF columns + chrom prefix.
        assert "chrom" in df.columns
        for col in ("start", "end", "period_size", "consensus_size", "score"):
            assert df[col].dtype.kind == "i"
        for col in ("copy_number", "pct_matches", "pct_indels", "entropy"):
            assert df[col].dtype.kind == "f"

    def test_chrom_assigned(self):
        df = parse_trf_dat(TRF_DAT)
        assert (df["chrom"] == "chr21").all()

    def test_start_is_zero_based(self):
        # A known TRF record: 1-based start 5016248 → 0-based 5016247.
        df = parse_trf_dat(TRF_DAT)
        rec = df[df["start"] == 5016247]
        assert len(rec) == 1
        assert rec.iloc[0]["end"] == 5016270

    def test_repeat_sequence_preserved(self):
        df = parse_trf_dat(TRF_DAT)
        mono_g = df[df["repeat_sequence"] == "GGGGGGGGGG"]
        assert len(mono_g) >= 1
        assert (mono_g["period_size"] == 1).all()

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_trf_dat(REPO_DATA / "no_such_file.dat")

    def test_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.dat"
        empty.write_text("Sequence: chrFOO\nParameters: 2 5 5 80 10 20 100\n\n")
        with pytest.raises(ValueError, match="No data lines"):
            parse_trf_dat(empty)


# ============================================================================
# ANNOTATE_REPEAT_ISOLATION
# ============================================================================

class TestAnnotateRepeatIsolation:
    def test_basic_columns_added(self):
        df = parse_trf_dat(TRF_DAT)
        ann = annotate_repeat_isolation(df, {"chr21": 50_000_000})
        for col in ("start_cover", "end_cover", "ldist", "rdist"):
            assert col in ann.columns

    def test_distances_non_negative(self):
        df = parse_trf_dat(TRF_DAT)
        ann = annotate_repeat_isolation(df, {"chr21": 50_000_000})
        assert (ann["ldist"] >= 0).all()
        assert (ann["rdist"] >= 0).all()

    def test_missing_chrom_raises(self):
        df = parse_trf_dat(TRF_DAT)
        with pytest.raises(KeyError):
            annotate_repeat_isolation(df, {"chrFOO": 1000})

    def test_overlapping_repeat_has_higher_coverage(self):
        # The fixture has a long repeat (row 8) that engulfs row 9.
        # After annotation, the engulfed repeat should have start_cover > 1.
        df = parse_trf_dat(TRF_DAT)
        ann = annotate_repeat_isolation(df, {"chr21": 50_000_000})
        # Row at start=5017756 sits inside the [5017612, 5017782) repeat.
        engulfed = ann[ann["start"] == 5017756]
        assert len(engulfed) == 1
        assert engulfed.iloc[0]["start_cover"] >= 2


# ============================================================================
# FILTER_ISOLATED_REPEATS
# ============================================================================

class TestFilterIsolatedRepeats:
    def _annotated(self):
        df = parse_trf_dat(TRF_DAT)
        return annotate_repeat_isolation(df, {"chr21": 50_000_000})

    def test_default_filtering_removes_overlaps(self):
        ann = self._annotated()
        out = filter_isolated_repeats(ann)
        # Engulfed repeat should be filtered out (start_cover != 1).
        assert (out["start"] != 5017756).all()

    def test_max_period_filter(self):
        ann = self._annotated()
        out = filter_isolated_repeats(ann, max_period=2)
        assert (out["period_size"] <= 2).all()

    def test_perfect_only_keeps_pure_mono_drops_impure(self):
        ann = self._annotated()
        out = filter_isolated_repeats(ann, perfect_only=True, min_dist=0)
        # period_size == 1 rows that survive must have a pure single-base sequence.
        mono = out[out["period_size"] == 1]
        for _, row in mono.iterrows():
            assert len(set(row["repeat_sequence"].upper())) == 1

    def test_require_isolated_off_keeps_overlaps(self):
        ann = self._annotated()
        on = filter_isolated_repeats(ann, require_isolated=True, min_dist=0)
        off = filter_isolated_repeats(ann, require_isolated=False, min_dist=0)
        assert len(off) >= len(on)

    def test_returns_a_copy(self):
        ann = self._annotated()
        out = filter_isolated_repeats(ann, min_dist=0)
        # Mutating the result must not affect the input.
        if len(out) > 0:
            out_index = out.index[0]
            out.loc[out_index, "start"] = -1
            assert ann.loc[out_index, "start"] != -1


# ============================================================================
# READ_CHROM_LENGTHS
# ============================================================================

class TestReadChromLengths:
    def test_reads_existing_fai(self, tmp_path):
        fa = tmp_path / "ref.fa"
        fai = tmp_path / "ref.fa.fai"
        fa.write_text(">contigA\nACGT\n")
        fai.write_text("contigA\t1234\t9\t60\t61\ncontigB\t5678\t100\t60\t61\n")
        out = read_chrom_lengths(fa)
        assert out == {"contigA": 1234, "contigB": 5678}

    def test_missing_fasta_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_chrom_lengths(tmp_path / "no_such.fa")
