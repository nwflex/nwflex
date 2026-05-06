"""
test_simulation.py — Tests for the simulation harness used by notebook 07
(NW-flex vs BWA-MEM comparison).
"""

import shutil

import pytest

from nwflex.repeats import STRLocus
from nwflex.simulation import (
    BwaBothStrandsResult,
    BwaResult,
    Haplotype,
    Read,
    _parse_sam_line,
    align_bwa,
    align_bwa_both_strands,
    build_haplotype,
    build_locus_from_panel,
    clean_flank_window,
    decode_z_bp,
    flank_bases_consumed,
    is_arm_correct,
    parse_cigar,
    rc_to_forward_alignment,
    render_zoom,
    reverse_complement,
    tile_reads,
)


_BWA_AVAILABLE = shutil.which("bwa") is not None
requires_bwa = pytest.mark.skipif(
    not _BWA_AVAILABLE, reason="requires `bwa` on PATH"
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


# ---------------------------------------------------------------------------
# reverse_complement
# ---------------------------------------------------------------------------

class TestReverseComplement:
    """reverse_complement — DNA reverse complement, preserving N."""

    def test_basic_palindrome(self):
        assert reverse_complement("ACGT") == "ACGT"

    def test_complement_and_reverse(self):
        assert reverse_complement("AAAGGG") == "CCCTTT"

    def test_n_passes_through(self):
        assert reverse_complement("ANCGN") == "NCGNT"

    def test_empty(self):
        assert reverse_complement("") == ""

    def test_lowercase_preserved(self):
        assert reverse_complement("acgt") == "acgt"


# ---------------------------------------------------------------------------
# _parse_sam_line
# ---------------------------------------------------------------------------

class TestParseSamLine:
    """_parse_sam_line — pull (read_id, BwaResult) from a SAM record."""

    def test_header_returns_none(self):
        assert _parse_sam_line("@HD\tVN:1.6\tSO:unsorted") is None
        assert _parse_sam_line("@SQ\tSN:locus\tLN:415") is None

    def test_short_line_returns_none(self):
        assert _parse_sam_line("not\ta\tsam\tline") is None

    def test_aligned_record_parses(self):
        # Minimal SAM record: read r0 aligned at pos 12 with CIGAR 5M, AS:i:5.
        line = (
            "r0\t0\tlocus\t12\t60\t5M\t*\t0\t0\t"
            "ACGTA\tIIIII\tNM:i:0\tAS:i:5\tXS:i:0"
        )
        item = _parse_sam_line(line)
        assert item is not None
        rid, result = item
        assert rid == "r0"
        assert result.is_unmapped is False
        assert result.pos == 12
        assert result.cigar == "5M"
        assert result.score == 5

    def test_unmapped_record_parses(self):
        # Flag 0x4 = unmapped; pos/cigar irrelevant per SAM spec.
        line = "r1\t4\t*\t0\t0\t*\t*\t0\t0\tACGTA\tIIIII"
        item = _parse_sam_line(line)
        assert item is not None
        rid, result = item
        assert rid == "r1"
        assert result.is_unmapped is True
        assert result.pos is None
        assert result.cigar is None
        assert result.score is None

    def test_secondary_alignment_skipped(self):
        # Flag 0x100 = secondary.
        line = "r0\t256\tlocus\t12\t60\t5M\t*\t0\t0\tACGTA\tIIIII\tAS:i:5"
        assert _parse_sam_line(line) is None

    def test_supplementary_alignment_skipped(self):
        # Flag 0x800 = supplementary.
        line = "r0\t2048\tlocus\t12\t60\t5M\t*\t0\t0\tACGTA\tIIIII\tAS:i:5"
        assert _parse_sam_line(line) is None


# ---------------------------------------------------------------------------
# align_bwa
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_locus_reads():
    """Build a tiny locus + haplotype + reads for BWA round-trip tests."""
    locus = build_locus_from_panel(
        motif="ACT",
        panel_lflank="GACGTGACGTGACGTGACGTGACGTGACGTGACGTGACGTGACGTGACGT",
        panel_rflank="CGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACG",
        ref_n=4,
        flank_len=40,
    )
    hap = build_haplotype(locus, delta=0)
    reads = tile_reads(hap, read_len=60, k_min_flank=4)
    return locus, hap, reads


@requires_bwa
class TestAlignBwa:
    """align_bwa — single-strand BWA-MEM round-trip."""

    def test_returns_one_result_per_read(self, tiny_locus_reads):
        locus, _, reads = tiny_locus_reads
        results = align_bwa(locus.X, reads, no_clip=False)
        assert len(results) == len(reads)
        assert all(isinstance(r, BwaResult) for r in results)

    def test_self_alignment_maps_every_read(self, tiny_locus_reads):
        # delta=0 reads are exact substrings of the reference; BWA must map.
        locus, _, reads = tiny_locus_reads
        results = align_bwa(locus.X, reads, no_clip=False)
        assert all(not r.is_unmapped for r in results)
        assert all(r.cigar is not None and r.score is not None for r in results)

    def test_no_clip_disables_soft_clipping(self):
        # Construct a read with mismatched flanks so BWA would normally
        # soft-clip; under -L 500 it should not emit `S` ops.
        locus = build_locus_from_panel(
            motif="ACT",
            panel_lflank="GACGTGACGTGACGTGACGTGACGTGACGTGACGTGACGTGACGT",
            panel_rflank="CGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT",
            ref_n=4,
            flank_len=40,
        )
        # Replace the first 5 bases of the read with garbage; standard BWA
        # tends to soft-clip such tails.
        reads = [Read(
            sequence="NNNNN" + locus.X[5:55],
            var_start=0, lflank_extent=40, rflank_extent=10,
        )]
        std = align_bwa(locus.X, reads, no_clip=False)
        nc = align_bwa(locus.X, reads, no_clip=True)
        # Whatever the standard arm did, the no-clip arm must not soft-clip.
        if not nc[0].is_unmapped:
            assert "S" not in nc[0].cigar
        # Sanity-check: the standard arm did get to choose freely.
        assert std[0] is not None


# ---------------------------------------------------------------------------
# align_bwa_both_strands
# ---------------------------------------------------------------------------

@requires_bwa
class TestAlignBwaBothStrands:
    """align_bwa_both_strands — both-orientation BWA-MEM."""

    def test_returns_one_result_per_read(self, tiny_locus_reads):
        locus, _, reads = tiny_locus_reads
        results = align_bwa_both_strands(locus.X, reads, no_clip=False)
        assert len(results) == len(reads)
        assert all(isinstance(r, BwaBothStrandsResult) for r in results)

    def test_self_alignment_maps_both_strands(self, tiny_locus_reads):
        locus, _, reads = tiny_locus_reads
        results = align_bwa_both_strands(locus.X, reads, no_clip=False)
        for r in results:
            assert not r.fwd.is_unmapped
            assert not r.rc.is_unmapped

    def test_strands_can_disagree_on_cigar(self, tiny_locus_reads):
        # We don't require disagreement, but the per-strand CIGARs must be
        # independently valid CIGAR strings.
        locus, _, reads = tiny_locus_reads
        results = align_bwa_both_strands(locus.X, reads, no_clip=False)
        for r in results:
            assert r.fwd.cigar and all(c.isdigit() or c in "MIDNSHP=X" for c in r.fwd.cigar)
            assert r.rc.cigar and all(c.isdigit() or c in "MIDNSHP=X" for c in r.rc.cigar)


# ---------------------------------------------------------------------------
# parse_cigar
# ---------------------------------------------------------------------------

class TestParseCigar:
    """parse_cigar — tokenize a CIGAR into (length, op) tuples."""

    def test_simple(self):
        assert parse_cigar("5M2I3M") == [(5, "M"), (2, "I"), (3, "M")]

    def test_all_ops_round_trip(self):
        cigar = "10S5=2X3I4D2N1H1P"
        parts = parse_cigar(cigar)
        assert "".join(f"{n}{op}" for n, op in parts) == cigar

    def test_single_op(self):
        assert parse_cigar("100M") == [(100, "M")]

    def test_empty_returns_empty(self):
        assert parse_cigar("") == []

    def test_malformed_raises(self):
        with pytest.raises(ValueError, match="invalid CIGAR"):
            parse_cigar("5M2Q3M")  # Q is not a CIGAR op
        with pytest.raises(ValueError, match="invalid CIGAR"):
            parse_cigar("5MM")  # missing length


# ---------------------------------------------------------------------------
# decode_z_bp
# ---------------------------------------------------------------------------

class TestDecodeZBp:
    """decode_z_bp — count read bp inside the repeat interval."""

    # Reference layout used in these tests:
    #   pos 0..9   left flank
    #   pos 10..19 repeat interval [10, 20)
    #   pos 20..29 right flank
    Z_START = 10
    Z_END = 20

    def test_centered_match_through_repeat(self):
        # 30M starting at 1-based pos 1 covers ref [0, 30): 10 bp inside Z.
        z = decode_z_bp("30M", 1, self.Z_START, self.Z_END, convention="bwa")
        assert z == 10

    def test_match_starting_inside_repeat(self):
        # 5M starting at pos 13 (0-based 12) covers [12, 17): 5 bp inside Z.
        z = decode_z_bp("5M", 13, self.Z_START, self.Z_END, convention="bwa")
        assert z == 5

    def test_match_outside_repeat_only(self):
        # 5M starting at pos 1 covers [0, 5): no overlap with [10, 20).
        z = decode_z_bp("5M", 1, self.Z_START, self.Z_END, convention="bwa")
        assert z == 0

    def test_deletion_inside_repeat_does_not_count(self):
        # 5M5D5M from pos 6 covers ref [5, 20). Ref bp [10, 15) are deleted
        # (no read bp); ref bp [15, 20) are matched (5 read bp inside Z).
        z = decode_z_bp("5M5D5M", 6, self.Z_START, self.Z_END, convention="bwa")
        assert z == 5

    def test_insertion_strictly_inside_counted_under_both_conventions(self):
        # 5M3I5M from pos 8 (ref_pos starts at 7):
        #   first 5M walks 7..11 → 2 inside Z (10, 11), ref_pos=12;
        #   3I at ref_pos=12 → 3 inside under both conventions;
        #   second 5M walks 12..16 → 5 inside Z, ref_pos=17.
        for conv in ("bwa", "nwflex"):
            z = decode_z_bp("5M3I5M", 8, self.Z_START, self.Z_END, convention=conv)
            assert z == 2 + 3 + 5

    def test_insertion_at_left_boundary_distinguishes_conventions(self):
        # 5M2I5M from pos 6: M walks 5..10, cursor at 10 during I, M walks 10..15.
        # ref_pos == z_start == 10 at insertion.
        # bwa convention: insertion counted inside (10 <= 10 <= 20).
        # nwflex convention: insertion counted outside (10 < 10 is false).
        bwa = decode_z_bp("5M2I5M", 6, self.Z_START, self.Z_END, convention="bwa")
        nwf = decode_z_bp("5M2I5M", 6, self.Z_START, self.Z_END, convention="nwflex")
        # Match contributions: 5M at 5..10 → 0 inside; 5M at 10..15 → 5 inside.
        assert bwa == 5 + 2
        assert nwf == 5

    def test_soft_clip_does_not_consume_reference(self):
        # 5S10M starting at pos 11 covers ref [10, 20): all 10 bp inside Z.
        # The soft-clip prefix doesn't affect ref_pos.
        z = decode_z_bp("5S10M", 11, self.Z_START, self.Z_END, convention="bwa")
        assert z == 10

    def test_invalid_convention_raises(self):
        with pytest.raises(ValueError, match="convention"):
            decode_z_bp("10M", 1, self.Z_START, self.Z_END, convention="other")


# ---------------------------------------------------------------------------
# flank_bases_consumed
# ---------------------------------------------------------------------------

class TestFlankBasesConsumed:
    """flank_bases_consumed — ref bp consumed in left and right flanks."""

    Z_START = 10
    Z_END = 20

    def test_alignment_spans_full_repeat(self):
        # 30M from pos 1 covers ref [0, 30): 10 left, 10 right.
        left, right = flank_bases_consumed("30M", 1, self.Z_START, self.Z_END)
        assert (left, right) == (10, 10)

    def test_alignment_inside_repeat_only(self):
        # 5M from pos 13 covers [12, 17): no flank coverage.
        left, right = flank_bases_consumed("5M", 13, self.Z_START, self.Z_END)
        assert (left, right) == (0, 0)

    def test_left_flank_only(self):
        # 8M from pos 1 covers [0, 8): 8 in left flank, 0 in repeat, 0 in right.
        left, right = flank_bases_consumed("8M", 1, self.Z_START, self.Z_END)
        assert (left, right) == (8, 0)

    def test_soft_clip_does_not_consume_ref(self):
        # 5S5M from pos 1 covers ref [0, 5): the S prefix doesn't add ref bp.
        left, right = flank_bases_consumed("5S5M", 1, self.Z_START, self.Z_END)
        assert (left, right) == (5, 0)

    def test_deletion_counts_as_consumed(self):
        # 5M5D5M from pos 1 covers ref [0, 15): 10 in left flank (incl D),
        # 5 in Z, 0 in right flank.
        left, right = flank_bases_consumed(
            "5M5D5M", 1, self.Z_START, self.Z_END
        )
        assert (left, right) == (10, 0)


# ---------------------------------------------------------------------------
# is_arm_correct
# ---------------------------------------------------------------------------

class TestIsArmCorrect:
    """is_arm_correct — z_bp matches truth AND the alignment spans the repeat."""

    Z_START = 10
    Z_END = 20

    def test_correct_when_truth_matches_and_flanks_covered(self):
        # 30M from pos 1: 10 inside Z, 10 left flank, 10 right flank.
        assert is_arm_correct(
            "30M", 1, self.Z_START, self.Z_END, truth_z_bp=10, convention="bwa"
        )

    def test_wrong_z_bp_is_incorrect(self):
        # 30M from pos 1 puts 10 bp inside Z, but truth=8.
        assert not is_arm_correct(
            "30M", 1, self.Z_START, self.Z_END, truth_z_bp=8, convention="bwa"
        )

    def test_no_left_flank_is_incorrect(self):
        # 5S20M from pos 11 puts 10 inside Z, 10 right, 0 left → fails span rule.
        assert not is_arm_correct(
            "5S20M", 11, self.Z_START, self.Z_END,
            truth_z_bp=10, convention="bwa",
        )

    def test_unmapped_is_incorrect(self):
        assert not is_arm_correct(
            None, None, self.Z_START, self.Z_END,
            truth_z_bp=10, convention="bwa",
        )

    def test_min_flank_threshold_enforced(self):
        # 30M from pos 1 covers 10 bp on each flank — passes min_flank=10.
        assert is_arm_correct(
            "30M", 1, self.Z_START, self.Z_END,
            truth_z_bp=10, convention="bwa", min_flank=10,
        )
        # min_flank=11 fails.
        assert not is_arm_correct(
            "30M", 1, self.Z_START, self.Z_END,
            truth_z_bp=10, convention="bwa", min_flank=11,
        )

    def test_convention_affects_boundary_insertion(self):
        # 1M2I11M from pos 10 (ref_pos starts at 9):
        #   1M consumes ref pos 9 (left flank, +1 ref bp);
        #   2I at ref_pos=10 — exactly z_start: bwa counts inside, nwflex outside;
        #   11M walks 10..20 — 10 bp inside Z plus 1 bp in the right flank.
        # bwa  → z_bp = 0 + 2 + 10 = 12; flanks (1, 1).
        # nwf  → z_bp = 0 + 0 + 10 = 10; flanks (1, 1).
        assert is_arm_correct(
            "1M2I11M", 10, self.Z_START, self.Z_END,
            truth_z_bp=12, convention="bwa",
        )
        assert not is_arm_correct(
            "1M2I11M", 10, self.Z_START, self.Z_END,
            truth_z_bp=12, convention="nwflex",
        )


# ---------------------------------------------------------------------------
# render_zoom
# ---------------------------------------------------------------------------

class TestRenderZoom:
    """render_zoom — column-aligned ASCII view of an alignment near the zone."""

    # 30 bp toy reference: 10 bp left flank | 10 bp repeat | 10 bp right flank.
    REF = "AAAAAAAAAA" + "GTGTGTGTGT" + "CCCCCCCCCC"
    Z_START = 10
    Z_END = 20

    def test_pure_match_through_zone(self):
        # 30M from pos 1 walks the full reference; the read equals the ref.
        out = render_zoom(self.REF, self.REF, 1, "30M", self.Z_START, self.Z_END)
        # Two lines (no I/D markers).
        assert out.splitlines() == [
            "ref :  AAAAAAAAAA|GTGTGTGTGT|CCCCCCCCCC",
            "read:  AAAAAAAAAA|GTGTGTGTGT|CCCCCCCCCC",
        ]

    def test_insertion_at_left_boundary_marks_dashes(self):
        # 10M3I20M from pos 1: 10M consumes left flank, 3I sits at ref_pos=10
        # (== z_start), 20M consumes the rest. Read carries 3 extra bases at
        # the boundary.
        read = "AAAAAAAAAA" + "GTG" + "GTGTGTGTGT" + "CCCCCCCCCC"
        out = render_zoom(self.REF, read, 1, "10M3I20M", self.Z_START, self.Z_END)
        lines = out.splitlines()
        assert lines[0] == "ref :  AAAAAAAAAA|---GTGTGTGTGT|CCCCCCCCCC"
        assert lines[1] == "read:  AAAAAAAAAA|GTGGTGTGTGTGT|CCCCCCCCCC"
        # The I-marker line must carry exactly three "I" columns under the gap.
        assert lines[2].count("I") == 3

    def test_right_soft_clip_is_bracketed(self):
        # 25M5S from pos 1: 25M consumes ref[0..25), then 5 read bases
        # are soft-clipped off the right end.
        read = self.REF[:25] + "TTTTT"
        out = render_zoom(self.REF, read, 1, "25M5S", self.Z_START, self.Z_END)
        ref_line, read_line = out.splitlines()[:2]
        assert "[TTTTT]" in read_line
        assert "[" not in ref_line

    def test_left_soft_clip_is_bracketed_before_alignment(self):
        # 5S25M from pos 6: 5 read bases clipped, then 25M consumes ref[5..30).
        read = "TTTTT" + self.REF[5:]
        out = render_zoom(self.REF, read, 6, "5S25M", self.Z_START, self.Z_END)
        read_line = out.splitlines()[1]
        # The clip prefix sits at the left edge of the alignment, bracketed.
        # (Line still has the "read:  " prefix; strip it before checking.)
        assert read_line.removeprefix("read:  ").startswith("[TTTTT]")

    def test_deletion_shows_dash_on_read(self):
        # 10M5D15M from pos 1: 10M consumes left flank, 5D removes ref[10..15)
        # without contributing read bases, 15M consumes ref[15..30).
        read = self.REF[:10] + self.REF[15:]
        out = render_zoom(self.REF, read, 1, "10M5D15M", self.Z_START, self.Z_END)
        lines = out.splitlines()
        # Five gap-on-read columns inside the zone.
        assert "-----" in lines[1]
        # Marker line carries five 'D's.
        assert lines[2].count("D") == 5

    def test_skip_op_marks_n_not_d(self):
        # 10M5N15M from pos 1: same column shape as a 5D, but the marker
        # must say 'N' (skipped reference, e.g. an EP-pattern jump in
        # NW-flex) — not 'D'.
        read = self.REF[:10] + self.REF[15:]
        out = render_zoom(self.REF, read, 1, "10M5N15M", self.Z_START, self.Z_END)
        lines = out.splitlines()
        assert "-----" in lines[1]
        assert lines[2].count("N") == 5
        assert "D" not in lines[2]

    def test_pad_trims_view(self):
        # With pad=2 the view should keep only 2 columns of flank on each
        # side of the zone (plus the two pipe separators).
        out = render_zoom(self.REF, self.REF, 1, "30M", self.Z_START, self.Z_END, pad=2)
        ref_line = out.splitlines()[0]
        # 7 chars of context + 'ref :  ' prefix; the visible alignment is
        # "AA|GTGTGTGTGT|CC".
        assert ref_line.endswith("AA|GTGTGTGTGT|CC")


# ---------------------------------------------------------------------------
# rc_to_forward_alignment
# ---------------------------------------------------------------------------

class TestRcToForwardAlignment:
    """rc_to_forward_alignment — flip an rc-strand hit to forward coords."""

    def test_simple_match_position(self):
        # Reference length 10. rc alignment "5M" at rc_pos=3 covers
        # rc_ref positions [3..7] (1-based) → forward ref [4..8].
        fwd_pos, fwd_cigar = rc_to_forward_alignment(3, "5M", 10)
        assert fwd_pos == 4
        assert fwd_cigar == "5M"

    def test_cigar_op_order_reverses(self):
        # 2M3I5M consumes 7 ref bases. Flipping reverses the op order.
        # fwd_pos = 20 - (2-1) - 7 + 1 = 13.
        fwd_pos, fwd_cigar = rc_to_forward_alignment(2, "2M3I5M", 20)
        assert fwd_pos == 13
        assert fwd_cigar == "5M3I2M"

    def test_left_soft_clip_becomes_right(self):
        # An S clip at the start of the rc read sits at the end of the
        # forward read; the flipped CIGAR moves it to the right end.
        _, fwd_cigar = rc_to_forward_alignment(1, "4S5M", 20)
        assert fwd_cigar == "5M4S"

    def test_right_soft_clip_becomes_left(self):
        _, fwd_cigar = rc_to_forward_alignment(1, "5M4S", 20)
        assert fwd_cigar == "4S5M"

    def test_round_trip_is_identity(self):
        # Flipping twice (with the same ref length) reproduces the input.
        rc_pos, rc_cigar, ref_len = 7, "3M2D2I8M", 30
        fwd_pos, fwd_cigar = rc_to_forward_alignment(rc_pos, rc_cigar, ref_len)
        rt_pos, rt_cigar = rc_to_forward_alignment(fwd_pos, fwd_cigar, ref_len)
        assert (rt_pos, rt_cigar) == (rc_pos, rc_cigar)

    def test_deletion_consumes_reference(self):
        # 5M2D5M consumes 12 ref bases. fwd_pos = 20 - 2 - 12 + 1 = 7.
        fwd_pos, fwd_cigar = rc_to_forward_alignment(3, "5M2D5M", 20)
        assert fwd_pos == 7
        assert fwd_cigar == "5M2D5M"
