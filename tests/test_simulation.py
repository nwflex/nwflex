"""
test_simulation.py — Tests for the simulation harness used by notebook 07
(NW-flex vs BWA-MEM comparison).
"""

import pytest

from nwflex.repeats import STRLocus
from nwflex.simulation import (
    Haplotype,
    Read,
    build_haplotype,
    build_locus_from_panel,
    clean_flank_window,
    tile_reads,
)


# ---------------------------------------------------------------------------
# clean_flank_window
# ---------------------------------------------------------------------------

class TestCleanFlankWindow:
    """clean_flank_window — slide window until the motif boundary is clean."""

    def test_left_natural_anchor_is_clean(self):
        # motif "AC": natural left anchor "TGGG" ends in 'G' (≠ 'C') → clean.
        window = clean_flank_window(
            "AC", "GGGTTTGGG", flank_len=4, side="left"
        )
        assert window == "TGGG"

    def test_left_anchor_needs_one_advance(self):
        # motif "AC": natural anchor "TGGC" ends in 'C' (would extend "AC" → "ACAC...").
        # Slide one bp left to "TTGG".
        window = clean_flank_window(
            "AC", "GGGTTTGGC", flank_len=4, side="left"
        )
        assert window == "TTGG"

    def test_right_natural_anchor_is_clean(self):
        # motif "AC": natural right anchor "TTTG" starts with 'T' (≠ 'A') → clean.
        window = clean_flank_window(
            "AC", "TTTGGGAAA", flank_len=4, side="right"
        )
        assert window == "TTTG"

    def test_right_anchor_needs_one_advance(self):
        # motif "AC": natural anchor "ATTT" starts with 'A' (would partial-extend).
        # Slide one bp right to "TTTG".
        window = clean_flank_window(
            "AC", "ATTTGGGAA", flank_len=4, side="right"
        )
        assert window == "TTTG"

    def test_left_flank_too_short_raises(self):
        with pytest.raises(ValueError, match="left flank too short"):
            clean_flank_window("AC", "GGG", flank_len=4, side="left")

    def test_right_flank_too_short_raises(self):
        with pytest.raises(ValueError, match="right flank too short"):
            clean_flank_window("AC", "GGG", flank_len=4, side="right")

    def test_unsatisfiable_within_advance_limit_raises(self):
        # Every length-4 window in "CCCCCCC" ends in 'C' — never clean.
        with pytest.raises(ValueError, match="could not be cleaned"):
            clean_flank_window(
                "AC", "CCCCCCC", flank_len=4, side="left", max_advance=2
            )

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError, match="side must be"):
            clean_flank_window(
                "AC", "GGGGGGG", flank_len=4, side="middle"
            )


# ---------------------------------------------------------------------------
# build_locus_from_panel
# ---------------------------------------------------------------------------

class TestBuildLocusFromPanel:
    """build_locus_from_panel — assemble a truth STRLocus from panel sequences."""

    def test_returns_strlocus_with_truth_n(self):
        locus = build_locus_from_panel(
            motif="AC",
            panel_lflank="GGGGGGGGGGT",
            panel_rflank="TGGGGGGGGGG",
            ref_n=5,
            flank_len=4,
        )
        assert isinstance(locus, STRLocus)
        assert locus.R == "AC"
        assert locus.N == 5
        assert len(locus.A) == 4
        assert len(locus.B) == 4

    def test_flanks_are_clean_at_motif_boundary(self):
        locus = build_locus_from_panel(
            motif="AC",
            panel_lflank="GGGGGGGGGGT",
            panel_rflank="TGGGGGGGGGG",
            ref_n=3,
            flank_len=4,
        )
        # Left flank's last base must not equal motif's last base.
        assert locus.A[-1] != "C"
        # Right flank's first base must not equal motif's first base.
        assert locus.B[0] != "A"

    def test_assembled_reference_is_consistent(self):
        locus = build_locus_from_panel(
            motif="AC",
            panel_lflank="GGGGGGGGGGT",
            panel_rflank="TGGGGGGGGGG",
            ref_n=3,
            flank_len=4,
        )
        assert locus.X == locus.A + "AC" * 3 + locus.B
        assert locus.n == 4 + 6 + 4

    def test_advances_when_natural_anchor_dirty(self):
        # Left panel ends "...GGC" with motif "AC": one advance needed.
        # Right panel starts "ATTT..." with motif "AC": one advance needed.
        locus = build_locus_from_panel(
            motif="AC",
            panel_lflank="GGGTTTGGC",
            panel_rflank="ATTTGGGAA",
            ref_n=2,
            flank_len=4,
        )
        assert locus.A == "TTGG"
        assert locus.B == "TTTG"


# ---------------------------------------------------------------------------
# build_haplotype
# ---------------------------------------------------------------------------

class TestBuildHaplotype:
    """build_haplotype — perturb repeat count, optionally apply one SNV."""

    @staticmethod
    def _locus():
        return STRLocus(A="GAGAG", R="ACT", N=4, B="GCGCG")

    def test_delta_zero_returns_locus_X(self):
        locus = self._locus()
        hap = build_haplotype(locus, delta=0)
        assert isinstance(hap, Haplotype)
        assert hap.sequence == locus.X
        assert hap.flank_len == 5
        assert hap.body_len == 12  # |R| * (N + 0) = 3 * 4
        assert hap.snv_pos is None
        assert hap.snv_base is None

    def test_positive_delta_extends_repeat(self):
        locus = self._locus()
        hap = build_haplotype(locus, delta=2)
        assert hap.sequence == "GAGAG" + "ACT" * 6 + "GCGCG"
        assert hap.body_len == 18

    def test_negative_delta_shortens_repeat(self):
        locus = self._locus()
        hap = build_haplotype(locus, delta=-1)
        assert hap.sequence == "GAGAG" + "ACT" * 3 + "GCGCG"
        assert hap.body_len == 9

    def test_repeat_count_below_zero_raises(self):
        locus = self._locus()
        with pytest.raises(ValueError, match="negative"):
            build_haplotype(locus, delta=-5)

    def test_snv_replaces_byte_at_abs_pos(self):
        locus = self._locus()
        hap = build_haplotype(locus, delta=0, snv=(4, "T"))
        # Position 4 is the last base of the left flank ("GAGA[G]" → "GAGA[T]").
        assert hap.sequence[4] == "T"
        assert hap.sequence[:4] == locus.A[:4]
        assert hap.sequence[5:] == locus.X[5:]
        assert hap.snv_pos == 4
        assert hap.snv_base == "T"

    def test_snv_inside_repeat_body_is_allowed(self):
        # The function does not restrict SNV placement.
        locus = self._locus()
        snv_pos = locus.s + 1  # one bp inside the repeat block
        hap = build_haplotype(locus, delta=0, snv=(snv_pos, "G"))
        assert hap.sequence[snv_pos] == "G"
        assert hap.snv_pos == snv_pos

    def test_snv_out_of_range_raises(self):
        locus = self._locus()
        with pytest.raises(ValueError, match="out of range"):
            build_haplotype(locus, delta=0, snv=(999, "T"))
        with pytest.raises(ValueError, match="out of range"):
            build_haplotype(locus, delta=0, snv=(-1, "T"))

    def test_snv_multibase_raises(self):
        locus = self._locus()
        with pytest.raises(ValueError, match="single character"):
            build_haplotype(locus, delta=0, snv=(0, "AC"))


# ---------------------------------------------------------------------------
# tile_reads
# ---------------------------------------------------------------------------

class TestTileReads:
    """tile_reads — tile reads across a haplotype with flank-context constraint."""

    @staticmethod
    def _hap(flank_len=10, body_len=4):
        # Distinguishable bases per region make extents easy to read.
        seq = "L" * flank_len + "M" * body_len + "R" * flank_len
        return Haplotype(sequence=seq, flank_len=flank_len, body_len=body_len)

    def test_tiles_every_valid_start(self):
        # flank=10, body=4, read_len=10, k=2 → s ∈ [6, 8].
        hap = self._hap()
        reads = tile_reads(hap, read_len=10, k_min_flank=2)
        assert [r.var_start for r in reads] == [6, 7, 8]
        assert all(len(r.sequence) == 10 for r in reads)
        assert all(isinstance(r, Read) for r in reads)

    def test_flank_extents_match_overlap(self):
        # Read at s=6, read_len=10 covers [6, 16): 4 bp left flank, 2 bp right flank.
        hap = self._hap()
        reads = tile_reads(hap, read_len=10, k_min_flank=2)
        first = reads[0]
        assert first.var_start == 6
        assert first.lflank_extent == 4
        assert first.rflank_extent == 2

    def test_single_read_when_read_len_at_minimum(self):
        # Minimum read_len = body_len + 2*k = 4 + 4 = 8.
        hap = self._hap()
        reads = tile_reads(hap, read_len=8, k_min_flank=2)
        assert len(reads) == 1
        assert reads[0].var_start == 8
        assert reads[0].lflank_extent == 2
        assert reads[0].rflank_extent == 2

    def test_no_reads_fit_raises(self):
        hap = self._hap()
        with pytest.raises(ValueError, match="no reads fit"):
            tile_reads(hap, read_len=7, k_min_flank=2)

    def test_step_returns_subset(self):
        hap = self._hap()
        reads = tile_reads(hap, read_len=10, k_min_flank=2, step=2)
        assert [r.var_start for r in reads] == [6, 8]

    def test_step_below_one_raises(self):
        hap = self._hap()
        with pytest.raises(ValueError, match="step"):
            tile_reads(hap, read_len=10, k_min_flank=2, step=0)
        with pytest.raises(ValueError, match="step"):
            tile_reads(hap, read_len=10, k_min_flank=2, step=-1)

    def test_clamps_to_haplotype_end(self):
        # Asymmetric flanks: left=10, body=4, right=2. seq len = 16.
        # Without clamp: s_max = 8; with clamp: 16 - 10 = 6 → only s=6.
        seq = "L" * 10 + "M" * 4 + "R" * 2
        hap = Haplotype(sequence=seq, flank_len=10, body_len=4)
        reads = tile_reads(hap, read_len=10, k_min_flank=2)
        assert [r.var_start for r in reads] == [6]
        assert len(reads[0].sequence) == 10
