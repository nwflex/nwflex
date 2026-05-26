"""
test_simulation.py — Tests for the simulation harness used by notebook 07
(NW-flex vs BWA-MEM comparison).
"""

import shutil

import pytest

from nwflex.repeats import STRLocus
from nwflex.simulation import (
    BWACompoundMethod,
    BwaBothStrandsResult,
    BwaResult,
    CompoundHaplotype,
    CompoundLocus,
    Haplotype,
    NWFlexCompoundMethod,
    NwflexResult,
    Read,
    _parse_sam_line,
    align_bwa,
    align_bwa_both_strands,
    align_nwflex,
    alignment_state_multi,
    build_compound_haplotype,
    build_compound_locus_from_panel,
    build_compound_mirror_frame,
    build_haplotype,
    build_locus_from_panel,
    build_mirror_frame,
    bwa_compound_truth_cigar,
    clean_flank_window,
    decode_z_bp,
    flank_bases_consumed,
    is_arm_correct,
    is_arm_correct_multi,
    mirror_reads,
    nwflex_compound_truth_cigar,
    parse_cigar,
    pivot_for_heatmap,
    plot_correctness_heatmap_rows,
    sweep,
    SweepVariant,
    project_alignment_to_ref,
    rc_cigar_to_forward,
    rc_to_forward_alignment,
    render_zoom,
    reverse_complement,
    score_alignment,
    state_to_glyph,
    tile_reads,
    wrap_methods_for_multizone_truth,
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

class TestMirrorReads:
    """mirror_reads — rc-flip a list of reads without touching the reference."""

    @staticmethod
    def _read(seq, var_start, lf, rf):
        return Read(sequence=seq, var_start=var_start,
                    lflank_extent=lf, rflank_extent=rf)

    def test_empty_list_returns_empty_list(self):
        assert mirror_reads([]) == []

    def test_sequence_is_reverse_complemented(self):
        r = self._read("ACGTACG", var_start=0, lf=2, rf=3)
        (out,) = mirror_reads([r])
        assert out.sequence == reverse_complement("ACGTACG")

    def test_extents_are_swapped_and_var_start_preserved(self):
        r = self._read("ACGT", var_start=12, lf=3, rf=5)
        (out,) = mirror_reads([r])
        assert out.lflank_extent == 5  # was rf
        assert out.rflank_extent == 3  # was lf
        assert out.var_start == 12

    def test_round_trip_is_identity(self):
        r = self._read("ACGTACG", var_start=7, lf=2, rf=4)
        (back,) = mirror_reads(mirror_reads([r]))
        assert back == r

    def test_matches_build_mirror_frame_reads(self):
        # mirror_reads must produce the same rc_reads as build_mirror_frame,
        # which is the contract that lets sweep cells reuse it per-haplotype.
        r1 = self._read("ACGTACGT", var_start=3, lf=2, rf=4)
        r2 = self._read("TTTGCCAA", var_start=9, lf=1, rf=7)
        _, rc_reads_full, _ = build_mirror_frame(
            "AAAACCCCGGGGTTTT", reads=[r1, r2], zone=(4, 12),
        )
        assert mirror_reads([r1, r2]) == rc_reads_full


class TestBuildMirrorFrame:
    """build_mirror_frame — mirror one or two references in one call."""

    @staticmethod
    def _read(seq, var_start, lf, rf):
        return Read(sequence=seq, var_start=var_start,
                    lflank_extent=lf, rflank_extent=rf)

    def test_returns_three_tuple_without_extras(self):
        out = build_mirror_frame("AACCGGTT", reads=[], zone=(2, 6))
        assert len(out) == 3
        rc_ref, rc_reads, rc_zone = out
        assert rc_ref == "AACCGGTT"  # palindrome
        assert rc_reads == []
        assert rc_zone == (2, 6)

    def test_zone_is_mirrored(self):
        # 8 bp reference, forward zone (1, 4) → mirror zone (8-4, 8-1) = (4, 7).
        _, _, rc_zone = build_mirror_frame("AAAACCCC", reads=[], zone=(1, 4))
        assert rc_zone == (4, 7)

    def test_reads_have_rc_sequence_and_swapped_extents(self):
        r = self._read("ACGT", var_start=12, lf=3, rf=5)
        _, rc_reads, _ = build_mirror_frame("AAAA", reads=[r], zone=(0, 1))
        (rc_r,) = rc_reads
        assert rc_r.sequence == "ACGT"  # palindrome
        assert rc_r.var_start == 12
        assert rc_r.lflank_extent == 5  # swapped with rf
        assert rc_r.rflank_extent == 3

    def test_returns_five_tuple_with_extras(self):
        out = build_mirror_frame(
            "AACCGGTT", reads=[], zone=(2, 6),
            extra_reference="GGGGAAAA", extra_zone=(4, 7),
        )
        assert len(out) == 5
        _, _, _, rc_extra, rc_extra_zone = out
        # extra_ref length 8, extra_zone (4, 7) → rc_extra_zone (1, 4).
        assert rc_extra == reverse_complement("GGGGAAAA")
        assert rc_extra_zone == (1, 4)

    def test_extras_do_not_affect_primary(self):
        primary_only = build_mirror_frame(
            "AACCGGTT", reads=[], zone=(2, 6),
        )
        with_extras = build_mirror_frame(
            "AACCGGTT", reads=[], zone=(2, 6),
            extra_reference="GGGGAAAA", extra_zone=(4, 7),
        )
        assert primary_only == with_extras[:3]

    def test_only_extra_reference_raises(self):
        with pytest.raises(ValueError, match="must be provided together"):
            build_mirror_frame(
                "AACCGGTT", reads=[], zone=(2, 6),
                extra_reference="GGGGAAAA",
            )

    def test_only_extra_zone_raises(self):
        with pytest.raises(ValueError, match="must be provided together"):
            build_mirror_frame(
                "AACCGGTT", reads=[], zone=(2, 6),
                extra_zone=(4, 7),
            )


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
# align_nwflex
# ---------------------------------------------------------------------------

class TestAlignNwflex:
    """align_nwflex — direction-agnostic NW-flex harness."""

    @pytest.fixture
    def nwflex_setup(self, tiny_locus_reads):
        from nwflex.default import get_default_scoring
        from nwflex.ep_patterns import build_EP_STR_phase
        from nwflex.repeats import STRLocus

        locus, _, reads = tiny_locus_reads
        # 3N extended reference for the EP-skip enumeration.
        nwflex_locus = STRLocus(
            A=locus.A, R=locus.R, N=3 * locus.N, B=locus.B,
        )
        ep = build_EP_STR_phase(
            nwflex_locus.n, nwflex_locus.s, nwflex_locus.e, nwflex_locus.k,
        )
        score_matrix, gap_open, gap_extend, a2i = get_default_scoring("bwa_mem")
        return dict(
            reads=reads,
            reference=nwflex_locus.X,
            extra_predecessors=ep,
            score_matrix=score_matrix,
            gap_open=gap_open,
            gap_extend=gap_extend,
            alphabet_to_index=a2i,
        )

    def test_returns_one_result_per_read(self, nwflex_setup):
        reads = nwflex_setup.pop("reads")
        out = align_nwflex(nwflex_setup.pop("reference"), reads, **nwflex_setup)
        assert len(out) == len(reads)
        assert all(isinstance(r, NwflexResult) for r in out)

    def test_pos_and_cigar_are_set(self, nwflex_setup):
        reads = nwflex_setup.pop("reads")
        out = align_nwflex(nwflex_setup.pop("reference"), reads, **nwflex_setup)
        for r in out:
            assert r.pos >= 1
            assert r.cigar
            assert all(c.isdigit() or c in "MIDNSHP=X"
                       for c in r.cigar)

    def test_score_matches_score_alignment(self, nwflex_setup):
        # NW-flex's reported score must equal `score_alignment` on its own
        # CIGAR under the same scheme — that is the contract of the global
        # score for a global aligner.
        from nwflex.simulation import score_alignment

        reads = nwflex_setup["reads"]
        sc_kw = dict(
            score_matrix=nwflex_setup["score_matrix"],
            gap_open=nwflex_setup["gap_open"],
            gap_extend=nwflex_setup["gap_extend"],
            alphabet_to_index=nwflex_setup["alphabet_to_index"],
        )
        out = align_nwflex(
            nwflex_setup["reference"], reads,
            extra_predecessors=nwflex_setup["extra_predecessors"],
            **sc_kw,
        )
        for r, hit in zip(reads, out):
            recomputed = score_alignment(
                r.sequence, nwflex_setup["reference"],
                hit.pos, hit.cigar, **sc_kw,
            )
            assert abs(hit.score - recomputed) < 1e-6

    def test_empty_reads_returns_empty_list(self, nwflex_setup):
        nwflex_setup.pop("reads")
        out = align_nwflex(nwflex_setup.pop("reference"), [], **nwflex_setup)
        assert out == []


# ---------------------------------------------------------------------------
# score_alignment
# ---------------------------------------------------------------------------

class TestScoreAlignment:
    """score_alignment — recompute NW affine-gap score from a CIGAR.

    Soft-clip ``S`` is charged the same affine-gap penalty as an insertion
    of the same length; ``N`` and ``H``/``P`` contribute nothing.
    """

    # match=+1, mismatch=-1, gap_open=-3, gap_extend=-1.
    SCORE_KW = dict(
        score_matrix=[[+1, -1, -1, -1],
                      [-1, +1, -1, -1],
                      [-1, -1, +1, -1],
                      [-1, -1, -1, +1]],
        gap_open=-3.0,
        gap_extend=-1.0,
        alphabet_to_index={"A": 0, "C": 1, "G": 2, "T": 3},
    )

    def _score(self, read, ref, cigar, pos=1):
        from nwflex.simulation import score_alignment
        return score_alignment(read, ref, pos, cigar, **self.SCORE_KW)

    def test_perfect_match(self):
        assert self._score("AAAA", "AAAA", "4M") == 4

    def test_mismatch_contributes_negative(self):
        # 2 matches + 2 mismatches = +2 - 2 = 0.
        assert self._score("AAAA", "AACC", "4M") == 0

    def test_insertion_gap_cost(self):
        # 2M3I2M with all matches: 4 + (-3 + 3*-1) = -2.
        assert self._score("AACCCAA", "AAAA", "2M3I2M") == -2

    def test_deletion_gap_cost(self):
        # 2M3D2M with all matches: 4 + (-3 + 3*-1) = -2.
        assert self._score("AAAA", "AACCCAA", "2M3D2M") == -2

    def test_soft_clip_charged_like_insertion(self):
        # 4M then 3S: 4 + (-3 + 3*-1) = -2.  Same gap cost as 3I.
        assert self._score("AAAACCC", "AAAA", "4M3S") == -2

    def test_leading_soft_clip_charged(self):
        # 3S then 4M: 4 + (-3 + 3*-1) = -2.
        assert self._score("CCCAAAA", "AAAA", "3S4M") == -2

    def test_leading_and_trailing_soft_clip_both_charged(self):
        # 2S4M2S: 4 + 2*(-3 + 2*-1) = -6.  Two distinct gap-open events.
        assert self._score("CCAAAACC", "AAAA", "2S4M2S") == -6

    def test_n_op_is_free(self):
        # 2M2N2M consumes ref through N for free: 4 matches → 4.
        assert self._score("AAAA", "AACCAA", "2M2N2M") == 4

    def test_hard_clip_and_padding_contribute_nothing(self):
        assert self._score("AAAA", "AAAA", "1H4M1H") == 4
        assert self._score("AAAA", "AAAA", "1P4M1P") == 4


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
# is_arm_correct_multi
# ---------------------------------------------------------------------------

class TestIsArmCorrectMulti:
    """is_arm_correct_multi — multi-zone variant.

    Outer-span check must work the same way regardless of whether
    ``ref_zones`` is listed in monotonic (fwd) order or non-monotonic
    (rc) order.  Per-block z_bp check is independent of zone ordering.
    """

    # Two zones: [10, 20) and [22, 30).  Outer flanks in fwd order are
    # ref < 10 and ref >= 30.
    FWD_ZONES = [(10, 20), (22, 30)]
    # Same two zones, but listed in non-monotonic order (rc convention
    # keeps the logical [block1, block2] order, which after rc-reversal
    # flips the address order).
    RC_ZONES  = [(22, 30), (10, 20)]
    TRUTH = (10, 8)  # block 1 has 10 read bp inside, block 2 has 8

    def test_fwd_zones_correct_when_flanks_covered(self):
        # 40M from pos 1: covers ref [0, 40).  Inside block1=10, block2=8
        # (8 ref bp in [22, 30), all matched).  Left flank = 10 bp,
        # right flank = 10 bp.
        per_block, ok = is_arm_correct_multi(
            "40M", 1, self.FWD_ZONES, self.TRUTH, convention="bwa",
        )
        assert per_block == [True, True]
        assert ok

    def test_rc_zones_correct_when_flanks_covered(self):
        # Same 40M alignment as above, but zones listed in rc order.
        # Outer endpoints must be derived as min/max so the span check
        # still asks for ref < 10 and ref >= 30.
        per_block, ok = is_arm_correct_multi(
            "40M", 1, self.RC_ZONES, (8, 10), convention="bwa",
        )
        assert per_block == [True, True]
        assert ok

    def test_rc_zones_missing_outer_left_flank_is_incorrect(self):
        # CIGAR starts at pos 11 (1-based) = ref idx 10: aligns inside
        # block 1 directly, with no ref bp consumed before the leftmost
        # outer boundary (ref < 10).  Even if per-block z_bp matches,
        # the outer span check must fail.  Old code used outer_s = 22
        # (first listed zone start) which would incorrectly count
        # ref bp in [10, 22) as "left flank" and pass.
        per_block, ok = is_arm_correct_multi(
            "10M2D8M10M", 11, self.RC_ZONES, (8, 10),
            convention="bwa",
        )
        # block1 (= rc_zones[0] = [22, 30)) gets 8 read bp; block2 (= rc_zones[1] = [10, 20)) gets 10 read bp.
        assert per_block == [True, True]
        assert not ok  # spans must fail: 0 ref bp consumed at ref < 10

    def test_fwd_zones_missing_outer_left_flank_is_incorrect(self):
        # Same alignment shape as above but with zones in fwd order.
        # outer_s = min = 10 in either ordering, so behaviour matches.
        per_block, ok = is_arm_correct_multi(
            "10M2D8M10M", 11, self.FWD_ZONES, self.TRUTH,
            convention="bwa",
        )
        assert per_block == [True, True]
        assert not ok


# ---------------------------------------------------------------------------
# build_compound_locus_from_panel / CompoundLocus
# ---------------------------------------------------------------------------

class TestBuildCompoundLocus:
    """build_compound_locus_from_panel — assemble a compound locus from
    a two-row panel selected by motif length, exposing both the BWA
    reference (X) and the NW-flex extended reference (X_ext) on one
    object.  The flanks A and B come from row 1 and row 2 respectively;
    the bridge M is the concatenation of M_left (cleaned against R1) and
    M_right (cleaned against R2)."""

    @staticmethod
    def _panel():
        import pandas as pd
        return pd.DataFrame({
            "type":   ["AC", "GTC"],
            "lflank": ["GGGGGGGGGGT", "TTTTTTTTTTA"],
            "rflank": ["TGGGGGGGGGG", "ATTTTTTTTTT"],
        })

    @staticmethod
    def _build(**overrides):
        kw = dict(
            motif1_len=2, motif2_len=3,
            ref_n1=3, ref_n2=2,
            bridge_n1=2, bridge_n2=2,
            flank_len=4,
        )
        kw.update(overrides)
        return build_compound_locus_from_panel(
            TestBuildCompoundLocus._panel(), **kw,
        )

    def test_returns_compoundlocus_with_panel_motifs(self):
        loc = self._build()
        assert isinstance(loc, CompoundLocus)
        assert loc.R1 == "AC" and loc.N1 == 3
        assert loc.R2 == "GTC" and loc.N2 == 2
        assert len(loc.A) == 4 and len(loc.B) == 4
        assert len(loc.M) == 2 + 2  # bridge_n1 + bridge_n2

    def test_flanks_and_bridge_are_clean_at_motif_boundaries(self):
        loc = self._build()
        # A must not partial-extend R1 rightward into the body
        # (cleaning side="left" requires A[-1] != R1[-1]).
        assert loc.A[-1] != loc.R1[-1]
        # B must not partial-extend R2 leftward into the body
        # (cleaning side="right" requires B[0] != R2[0]).
        assert loc.B[0] != loc.R2[0]
        # Bridge halves are cleaned independently:
        #   M_left  (cleaned as a "right" anchor of R1) → M_left[0]  != R1[0]
        #   M_right (cleaned as a "left"  anchor of R2) → M_right[-1] != R2[-1]
        assert loc.M[0] != loc.R1[0]
        assert loc.M[-1] != loc.R2[-1]

    def test_reference_sequence_matches_field_concatenation(self):
        loc = self._build()
        assert loc.X == loc.A + loc.R1 * loc.N1 + loc.M + loc.R2 * loc.N2 + loc.B

    def test_zones_align_with_motif_repeat_blocks_in_reference(self):
        loc = self._build()
        (s1, e1), (s2, e2) = loc.zones
        # Block 1 sits between A and the bridge; block 2 between bridge and B.
        assert s1 == len(loc.A)
        assert e1 == s1 + loc.k1 * loc.N1
        assert s2 == e1 + len(loc.M)
        assert e2 == s2 + loc.k2 * loc.N2
        # The reference bases inside each zone are pure repeats of the motif —
        # contract that downstream z_bp arithmetic relies on.
        assert loc.X[s1:e1] == loc.R1 * loc.N1
        assert loc.X[s2:e2] == loc.R2 * loc.N2

    def test_extended_reference_scales_each_block_by_nwflex_factor(self):
        loc = self._build(nwflex_factor=3)
        # X_ext keeps the same flanks and bridge but extends each repeat
        # block to f*N copies of its motif (so the EP-pattern can match
        # haplotype counts both below and above N).
        assert loc.X_ext == (
            loc.A + loc.R1 * (3 * loc.N1) + loc.M
            + loc.R2 * (3 * loc.N2) + loc.B
        )
        (s1, e1), (s2, e2) = loc.zones_ext
        assert (e1 - s1) == loc.k1 * 3 * loc.N1
        assert (e2 - s2) == loc.k2 * 3 * loc.N2
        assert loc.X_ext[s1:e1] == loc.R1 * (3 * loc.N1)
        assert loc.X_ext[s2:e2] == loc.R2 * (3 * loc.N2)


# ---------------------------------------------------------------------------
# build_compound_haplotype / CompoundHaplotype
# ---------------------------------------------------------------------------

class TestBuildCompoundHaplotype:
    """build_compound_haplotype — perturb both repeat counts independently
    and optionally apply one SNV.  Body length is the sum of both
    perturbed blocks plus the bridge; the flank length is taken straight
    from the locus."""

    @staticmethod
    def _locus():
        return CompoundLocus(
            A="GAGAG", R1="AC", N1=3,
            M="TT", R2="GTC", N2=2, B="GCGCG",
        )

    def test_zero_deltas_return_locus_reference(self):
        loc = self._locus()
        hap = build_compound_haplotype(loc, delta1=0, delta2=0)
        assert isinstance(hap, CompoundHaplotype)
        assert hap.sequence == loc.X
        assert hap.delta1 == 0 and hap.delta2 == 0
        assert hap.body_lens == (loc.k1 * loc.N1, loc.k2 * loc.N2)
        assert hap.body_len == loc.k1 * loc.N1 + len(loc.M) + loc.k2 * loc.N2
        assert hap.flank_len == len(loc.A)
        assert hap.snv_pos is None and hap.snv_base is None

    def test_positive_delta1_extends_only_block_one(self):
        loc = self._locus()
        hap = build_compound_haplotype(loc, delta1=2, delta2=0)
        assert hap.sequence == "GAGAG" + "AC" * 5 + "TT" + "GTC" * 2 + "GCGCG"
        assert hap.body_lens == (10, 6)  # (k1*(N1+d1), k2*N2)
        assert hap.body_len == 10 + len(loc.M) + 6

    def test_negative_delta2_shortens_only_block_two(self):
        loc = self._locus()
        hap = build_compound_haplotype(loc, delta1=0, delta2=-1)
        assert hap.sequence == "GAGAG" + "AC" * 3 + "TT" + "GTC" * 1 + "GCGCG"
        assert hap.body_lens == (6, 3)

    def test_both_deltas_compose_independently(self):
        loc = self._locus()
        hap = build_compound_haplotype(loc, delta1=1, delta2=-1)
        assert hap.sequence == "GAGAG" + "AC" * 4 + "TT" + "GTC" * 1 + "GCGCG"
        assert hap.body_lens == (8, 3)
        assert hap.delta1 == 1 and hap.delta2 == -1

    def test_block_one_count_below_zero_raises(self):
        loc = self._locus()
        with pytest.raises(ValueError, match=">= 0"):
            build_compound_haplotype(loc, delta1=-loc.N1 - 1, delta2=0)

    def test_block_two_count_below_zero_raises(self):
        loc = self._locus()
        with pytest.raises(ValueError, match=">= 0"):
            build_compound_haplotype(loc, delta1=0, delta2=-loc.N2 - 1)

    def test_snv_replaces_byte_at_abs_pos(self):
        loc = self._locus()
        # Position 4 is the last base of the left flank ("GAGA[G]" → "GAGA[T]").
        hap = build_compound_haplotype(loc, delta1=0, delta2=0, snv=(4, "T"))
        assert hap.sequence[4] == "T"
        assert hap.sequence[:4] == loc.A[:4]
        assert hap.sequence[5:] == loc.X[5:]
        assert hap.snv_pos == 4 and hap.snv_base == "T"

    def test_snv_inside_bridge_is_allowed(self):
        # The bridge sits at [flank_len + k1*N1, flank_len + k1*N1 + |M|).
        # build_compound_haplotype does not restrict SNV placement, so any
        # in-range index is accepted — including bases inside the bridge.
        loc = self._locus()
        bridge_start = len(loc.A) + loc.k1 * loc.N1
        hap = build_compound_haplotype(
            loc, delta1=0, delta2=0, snv=(bridge_start, "A"),
        )
        assert hap.sequence[bridge_start] == "A"

    def test_snv_out_of_range_raises(self):
        loc = self._locus()
        with pytest.raises(ValueError, match="out of range"):
            build_compound_haplotype(loc, delta1=0, delta2=0, snv=(999, "T"))
        with pytest.raises(ValueError, match="out of range"):
            build_compound_haplotype(loc, delta1=0, delta2=0, snv=(-1, "T"))

    def test_snv_multibase_raises(self):
        loc = self._locus()
        with pytest.raises(ValueError, match="single character"):
            build_compound_haplotype(loc, delta1=0, delta2=0, snv=(0, "AC"))


# ---------------------------------------------------------------------------
# build_compound_mirror_frame
# ---------------------------------------------------------------------------

class TestBuildCompoundMirrorFrame:
    """build_compound_mirror_frame — rc both references and both zone
    pairs.  The listing keeps [block1, block2] order (so per-block
    ground truth stays index-aligned across frames) even though the
    rc reversal flips the address order."""

    @staticmethod
    def _locus():
        return CompoundLocus(
            A="GAGAG", R1="AC", N1=3,
            M="TT", R2="GTC", N2=2, B="GCGCG",
        )

    @staticmethod
    def _read(seq, var_start, lf, rf):
        return Read(sequence=seq, var_start=var_start,
                    lflank_extent=lf, rflank_extent=rf)

    def test_returns_five_tuple(self):
        loc = self._locus()
        out = build_compound_mirror_frame(loc, reads=[])
        assert len(out) == 5

    def test_rc_references_are_reverse_complement_of_originals(self):
        loc = self._locus()
        rc_X, rc_X_ext, _, _, _ = build_compound_mirror_frame(loc, reads=[])
        assert rc_X == reverse_complement(loc.X)
        assert rc_X_ext == reverse_complement(loc.X_ext)

    def test_rc_reads_match_mirror_reads_of_input(self):
        # The compound mirror frame uses mirror_reads for the read flip,
        # which is the contract that lets downstream sweep cells reuse
        # mirror_reads per-haplotype without rebuilding the frame.
        loc = self._locus()
        r = self._read("ACGT", var_start=7, lf=2, rf=1)
        _, _, rc_reads, _, _ = build_compound_mirror_frame(loc, reads=[r])
        assert rc_reads == mirror_reads([r])

    def test_rc_zones_preserve_block_order_and_reverse_addresses(self):
        loc = self._locus()
        n = len(loc.X)
        _, _, _, rc_zones, _ = build_compound_mirror_frame(loc, reads=[])
        # rc_zones[i] is the rc of fwd zones[i], NOT zones[::-1] — the
        # block-i index must keep pointing to the same logical block.
        assert rc_zones == [(n - e, n - s) for (s, e) in loc.zones]
        # As a consequence, the address order is now non-monotonic:
        # rc block 1 (formerly leftmost in fwd) sits to the RIGHT of rc
        # block 2 in the rc reference.
        assert rc_zones[0][0] > rc_zones[1][0]

    def test_rc_zones_ext_mirror_extended_addresses(self):
        loc = self._locus()
        n_ext = len(loc.X_ext)
        _, _, _, _, rc_zones_ext = build_compound_mirror_frame(loc, reads=[])
        assert rc_zones_ext == [
            (n_ext - e, n_ext - s) for (s, e) in loc.zones_ext
        ]

    def test_rc_zone_bases_are_rc_of_forward_zone_bases(self):
        # Slicing rc_X at rc_zones[i] must yield the reverse-complement of
        # X[zones[i]] — i.e. the same motif content from the opposite
        # strand.  Sanity check that the rc index arithmetic lines up.
        loc = self._locus()
        rc_X, _, _, rc_zones, _ = build_compound_mirror_frame(loc, reads=[])
        for (s, e), (rs, re) in zip(loc.zones, rc_zones):
            assert rc_X[rs:re] == reverse_complement(loc.X[s:e])


# ---------------------------------------------------------------------------
# alignment_state_multi
# ---------------------------------------------------------------------------

class TestAlignmentStateMulti:
    """alignment_state_multi — compound P/T/M/D verdict.

    P (pass) requires both per-block z_bp matches AND the alignment
    spans both outer flanks (length axis only).  When the length axis
    fails, the score axis distinguishes M (chosen < truth — heuristic
    miss), T (chosen == truth — co-optimal tie-break), and D (chosen >
    truth — score landscape preferred a wrong alignment).  An unmapped
    or no-score alignment short-circuits to D.
    """

    # Two zones used across tests: block1 = [10, 20), block2 = [22, 30).
    FWD_ZONES = [(10, 20), (22, 30)]
    TRUTH     = (10, 8)  # block1 carries 10 read bp, block2 carries 8

    def test_pass_when_both_blocks_length_correct_with_flanks(self):
        # 40M from pos 1 covers ref [0, 40): per-block z_bp matches truth
        # (10 and 8), left flank = 10, right flank = 10.  Score axis is
        # irrelevant once length passes — pin chosen != truth to prove it.
        assert alignment_state_multi(
            "40M", 1, chosen_score=42, truth_score=99,
            ref_zones=self.FWD_ZONES, truth_z_bps=self.TRUTH,
            convention="bwa",
        ) == "P"

    def test_tied_when_length_wrong_and_score_equals_truth(self):
        # 30M from pos 11 skips the left flank entirely: per-block
        # z_bp still matches but the outer span check fails on the left.
        # With chosen_score == truth_score the state is T (co-optimal).
        state = alignment_state_multi(
            "30M", 11, chosen_score=50.0, truth_score=50.0,
            ref_zones=self.FWD_ZONES, truth_z_bps=self.TRUTH,
            convention="bwa",
        )
        assert state == "T"

    def test_missed_when_length_wrong_and_score_below_truth(self):
        state = alignment_state_multi(
            "30M", 11, chosen_score=40.0, truth_score=50.0,
            ref_zones=self.FWD_ZONES, truth_z_bps=self.TRUTH,
            convention="bwa",
        )
        assert state == "M"

    def test_dominated_when_length_wrong_and_score_above_truth(self):
        state = alignment_state_multi(
            "30M", 11, chosen_score=60.0, truth_score=50.0,
            ref_zones=self.FWD_ZONES, truth_z_bps=self.TRUTH,
            convention="bwa",
        )
        assert state == "D"

    def test_one_block_correct_other_not_falls_to_score_axis(self):
        # 40M with proper flank spans, but only block 1 z_bp matches —
        # block 2's truth_z_bps entry deliberately disagrees with the 8 bp
        # the alignment placed there.  all(per_block) is False, so the
        # state must fall to the score axis instead of returning P.
        state = alignment_state_multi(
            "40M", 1, chosen_score=50.0, truth_score=50.0,
            ref_zones=self.FWD_ZONES, truth_z_bps=(10, 7),
            convention="bwa",
        )
        assert state == "T"

    def test_unmapped_alignment_is_dominated(self):
        # No CIGAR / no pos / no score → no informative comparison; the
        # function must short-circuit to D.
        state = alignment_state_multi(
            None, None, chosen_score=None, truth_score=50.0,
            ref_zones=self.FWD_ZONES, truth_z_bps=self.TRUTH,
            convention="bwa",
        )
        assert state == "D"


# ---------------------------------------------------------------------------
# bwa_compound_truth_cigar
# ---------------------------------------------------------------------------

class TestBwaCompoundTruthCigar:
    """bwa_compound_truth_cigar — compound BWA ground-truth CIGAR with
    gaps placed at the LEFT edge of each repeat block (matching the BWA
    boundary convention).  Verify the emitted CIGAR has the expected
    shape, scores to its hand-tallied affine-gap value under
    score_alignment, and passes is_arm_correct_multi on the zones it was
    built for.
    """

    # match=+1, mismatch=-1, gap_open=-3, gap_extend=-1 (stand-alone open).
    SCORE_KW = dict(
        score_matrix=[[+1, -1, -1, -1],
                      [-1, +1, -1, -1],
                      [-1, -1, +1, -1],
                      [-1, -1, -1, +1]],
        gap_open=-3.0,
        gap_extend=-1.0,
        alphabet_to_index={"A": 0, "C": 1, "G": 2, "T": 3},
    )

    @staticmethod
    def _locus():
        # Small constructed compound: |A|=4, k1=2, N1=3, |M|=2, k2=2, N2=2,
        # |B|=4 → |X| = 4 + 6 + 2 + 4 + 4 = 20.
        return CompoundLocus(
            A="AAAA", R1="AC", N1=3,
            M="TT", R2="GT", N2=2, B="GGGG",
        )

    @staticmethod
    def _full_read(hap):
        # Read covers the entire haplotype with a 4 bp flank on each side.
        return Read(
            sequence=hap.sequence, var_start=0,
            lflank_extent=4, rflank_extent=4,
        )

    def test_zero_delta_truth_is_pure_match(self):
        loc = self._locus()
        hap = build_compound_haplotype(loc, delta1=0, delta2=0)
        pos, cigar = bwa_compound_truth_cigar(self._full_read(hap), hap, loc)
        # All ops are M and adjacent — they collapse to a single run.
        assert pos == 1
        assert cigar == f"{len(loc.X)}M"

    def test_positive_delta1_inserts_at_left_edge_of_block_one(self):
        loc = self._locus()
        hap = build_compound_haplotype(loc, delta1=1, delta2=0)
        pos, cigar = bwa_compound_truth_cigar(self._full_read(hap), hap, loc)
        # L=4 then 2I at the block-1 left edge, then everything else
        # (6 + 2 + 4 + 4) merges into one 16M run.
        assert pos == 1
        assert cigar == "4M2I16M"

    def test_negative_delta2_deletes_at_left_edge_of_block_two(self):
        loc = self._locus()
        hap = build_compound_haplotype(loc, delta1=0, delta2=-1)
        pos, cigar = bwa_compound_truth_cigar(self._full_read(hap), hap, loc)
        # L + block1 + bridge merge to 12M, then 2D, then 2M (body2) + 4M (Rf)
        # merge to 6M.
        assert pos == 1
        assert cigar == "12M2D6M"

    def test_both_block_deltas_compose(self):
        loc = self._locus()
        hap = build_compound_haplotype(loc, delta1=1, delta2=-1)
        pos, cigar = bwa_compound_truth_cigar(self._full_read(hap), hap, loc)
        # 4M | 2I | (6+2=8)M | 2D | (2+4=6)M.
        assert pos == 1
        assert cigar == "4M2I8M2D6M"

    def test_score_matches_handcalc_for_positive_delta1(self):
        loc = self._locus()
        hap = build_compound_haplotype(loc, delta1=1, delta2=0)
        read = self._full_read(hap)
        pos, cigar = bwa_compound_truth_cigar(read, hap, loc)
        # All match-aligned bases match by construction: 4 + 16 = 20.
        # One stand-alone gap of length 2 costs gap_open + 2*gap_extend = -5.
        expected = 20 - 5
        assert score_alignment(read.sequence, loc.X, pos, cigar,
                               **self.SCORE_KW) == expected

    def test_score_matches_handcalc_for_negative_delta2(self):
        loc = self._locus()
        hap = build_compound_haplotype(loc, delta1=0, delta2=-1)
        read = self._full_read(hap)
        pos, cigar = bwa_compound_truth_cigar(read, hap, loc)
        # 12 + 6 = 18 matches; one length-2 gap → 18 - 5 = 13.
        assert score_alignment(read.sequence, loc.X, pos, cigar,
                               **self.SCORE_KW) == 13

    def test_score_matches_handcalc_for_both_deltas(self):
        loc = self._locus()
        hap = build_compound_haplotype(loc, delta1=1, delta2=-1)
        read = self._full_read(hap)
        pos, cigar = bwa_compound_truth_cigar(read, hap, loc)
        # 4 + 8 + 6 = 18 matches; two distinct length-2 gaps → 18 - 10 = 8.
        assert score_alignment(read.sequence, loc.X, pos, cigar,
                               **self.SCORE_KW) == 8

    def test_truth_is_per_block_length_correct(self):
        # Sanity contract: a truth CIGAR must always pass is_arm_correct_multi
        # on the zones it was constructed for, regardless of delta sign.
        loc = self._locus()
        for d1, d2 in [(0, 0), (1, 0), (0, -1), (1, -1), (-1, 1)]:
            hap = build_compound_haplotype(loc, delta1=d1, delta2=d2)
            read = self._full_read(hap)
            pos, cigar = bwa_compound_truth_cigar(read, hap, loc)
            truth = (loc.k1 * (loc.N1 + d1), loc.k2 * (loc.N2 + d2))
            per_block, ok = is_arm_correct_multi(
                cigar, pos, loc.zones, truth, convention="bwa",
            )
            assert per_block == [True, True], (d1, d2, per_block)
            assert ok, (d1, d2)


# ---------------------------------------------------------------------------
# nwflex_compound_truth_cigar
# ---------------------------------------------------------------------------

class TestNwflexCompoundTruthCigar:
    """nwflex_compound_truth_cigar — compound NW-flex ground-truth CIGAR
    against the extended (f·N1, f·N2) reference.  An N op free-skips
    past the unused motifs at the LEFT edge of each repeat block, then
    M walks the haplotype's actual block bases.  No I or D ops are
    produced — the truth is structurally gap-free in NW-flex's frame.
    """

    SCORE_KW = TestBwaCompoundTruthCigar.SCORE_KW

    @staticmethod
    def _locus():
        # Same compound as the BWA truth tests, with f=3 so the extended
        # reference holds 9 R1 copies and 6 R2 copies.
        return CompoundLocus(
            A="AAAA", R1="AC", N1=3,
            M="TT", R2="GT", N2=2, B="GGGG",
            nwflex_factor=3,
        )

    @staticmethod
    def _full_read(hap):
        return Read(
            sequence=hap.sequence, var_start=0,
            lflank_extent=4, rflank_extent=4,
        )

    def test_zero_delta_skips_unused_motif_copies(self):
        loc = self._locus()
        hap = build_compound_haplotype(loc, delta1=0, delta2=0)
        pos, cigar = nwflex_compound_truth_cigar(self._full_read(hap), hap, loc)
        # skip1 = (f*N1 - N1)*k1 = (9-3)*2 = 12; body1 = 6.
        # skip2 = (f*N2 - N2)*k2 = (6-2)*2 = 8;  body2 = 4.
        # body1 + bridge merge to 8M; body2 + Rf merge to 8M.
        assert pos == 1
        assert cigar == "4M12N8M8N8M"

    def test_positive_delta1_shrinks_skip1_and_grows_body1(self):
        loc = self._locus()
        hap = build_compound_haplotype(loc, delta1=1, delta2=0)
        pos, cigar = nwflex_compound_truth_cigar(self._full_read(hap), hap, loc)
        # skip1 = (9-4)*2 = 10; body1 = 8. skip2 and body2 unchanged.
        assert cigar == "4M10N10M8N8M"

    def test_negative_delta2_grows_skip2_and_shrinks_body2(self):
        loc = self._locus()
        hap = build_compound_haplotype(loc, delta1=0, delta2=-1)
        pos, cigar = nwflex_compound_truth_cigar(self._full_read(hap), hap, loc)
        # skip2 = (6-1)*2 = 10; body2 = 2.
        assert cigar == "4M12N8M10N6M"

    def test_both_block_deltas_compose(self):
        loc = self._locus()
        hap = build_compound_haplotype(loc, delta1=1, delta2=-1)
        pos, cigar = nwflex_compound_truth_cigar(self._full_read(hap), hap, loc)
        assert cigar == "4M10N10M10N6M"

    def test_score_is_pure_match_count_with_no_gap_penalty(self):
        # N ops are free under score_alignment; the truth CIGAR carries
        # no I or D, so the score collapses to the total number of M bases
        # — all of which match by construction.  This proves the function
        # has not silently introduced a gap-shaped op.
        loc = self._locus()
        for d1, d2, expected_matches in [
            (0, 0, 20), (1, 0, 22), (0, -1, 18), (1, -1, 20),
        ]:
            hap = build_compound_haplotype(loc, delta1=d1, delta2=d2)
            read = self._full_read(hap)
            pos, cigar = nwflex_compound_truth_cigar(read, hap, loc)
            score = score_alignment(
                read.sequence, loc.X_ext, pos, cigar, **self.SCORE_KW,
            )
            assert score == expected_matches, (
                f"delta=({d1},{d2}): expected {expected_matches}, got {score}"
            )

    def test_truth_is_per_block_length_correct_on_extended_ref(self):
        # The N-skip placement must put the M walk exactly on the
        # haplotype's body bases inside each extended zone, so decode_z_bp
        # (nwflex convention) recovers (N+d)*k for each block.
        loc = self._locus()
        for d1, d2 in [(0, 0), (1, 0), (0, -1), (1, -1)]:
            hap = build_compound_haplotype(loc, delta1=d1, delta2=d2)
            read = self._full_read(hap)
            pos, cigar = nwflex_compound_truth_cigar(read, hap, loc)
            truth = (loc.k1 * (loc.N1 + d1), loc.k2 * (loc.N2 + d2))
            per_block, ok = is_arm_correct_multi(
                cigar, pos, loc.zones_ext, truth, convention="nwflex",
            )
            assert per_block == [True, True], (d1, d2, per_block)
            assert ok, (d1, d2)

    def test_haplotype_counts_exceeding_extended_ref_raise(self):
        # With factor=3, block 1 can hold at most f*N1 = 9 motif copies.
        # A haplotype with delta1=7 (count=10) would require a negative
        # skip — the function must raise rather than emit a malformed CIGAR.
        loc = self._locus()
        hap = build_compound_haplotype(loc, delta1=7, delta2=0)
        with pytest.raises(ValueError, match="extended ref"):
            nwflex_compound_truth_cigar(self._full_read(hap), hap, loc)


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
# project_alignment_to_ref
# ---------------------------------------------------------------------------

class TestProjectAlignmentToRef:
    """project_alignment_to_ref — CIGAR onto reference coordinates."""

    def test_pure_match_fills_full_span(self):
        # 5M at pos 1 against a 5 bp ref: each ref position carries the
        # corresponding read base.
        out = project_alignment_to_ref(5, 1, "5M", "GGGGG")
        assert out == "GGGGG"

    def test_offset_pos_leaves_bg_before_alignment(self):
        # 3M at pos 3 against a 6 bp ref: positions 0 and 1 are off-span,
        # positions 2..4 carry read bases, position 5 is off-span.
        out = project_alignment_to_ref(6, 3, "3M", "ACG", bg_char=".")
        assert out == "..ACG."

    def test_deletion_writes_gap_char(self):
        # 3M2D2M at pos 1 against a 7 bp ref: positions 0..2 carry read,
        # 3..4 are gaps (D), 5..6 carry read.
        out = project_alignment_to_ref(
            7, 1, "3M2D2M", "ACGCT", gap_char="-", bg_char=".",
        )
        assert out == "ACG--CT"

    def test_insertion_consumes_read_only(self):
        # 3M2I2M at pos 1 against a 5 bp ref: I consumes 2 read bases but
        # no ref positions; the projection is just the 3M + 2M bases.
        out = project_alignment_to_ref(5, 1, "3M2I2M", "ACGXXTT")
        assert out == "ACGTT"

    def test_soft_clip_does_not_appear(self):
        # 2S3M2S at pos 1 against a 3 bp ref: only the 3M positions are
        # filled; the soft-clipped read bases are consumed silently.
        out = project_alignment_to_ref(
            3, 1, "2S3M2S", "XXACGYY", bg_char=".",
        )
        assert out == "ACG"

    def test_hard_clip_consumes_nothing(self):
        # 2H3M at pos 1 against a 3 bp ref: H consumes neither ref nor
        # read; the 3M takes read positions 0..2 (read_seq has no
        # hard-clipped bases per SAM convention).
        out = project_alignment_to_ref(3, 1, "2H3M", "ACG")
        assert out == "ACG"

    def test_skip_op_writes_gap_char(self):
        # 2M3N2M at pos 1: skipped ref behaves like a delete for the
        # projection (a gap on the read line over those ref columns).
        out = project_alignment_to_ref(
            7, 1, "2M3N2M", "ACGT", gap_char="-", bg_char=".",
        )
        assert out == "AC---GT"

    def test_unsupported_op_raises(self):
        with pytest.raises(ValueError, match="unsupported CIGAR op"):
            project_alignment_to_ref(5, 1, "5P", "AAAAA")


# ---------------------------------------------------------------------------
# rc_to_forward_alignment
# ---------------------------------------------------------------------------

class TestRcCigarToForward:
    """rc_cigar_to_forward — return only the forward-strand CIGAR."""

    def test_matches_rc_to_forward_alignment_cigar(self):
        # Should return the same CIGAR rc_to_forward_alignment computes,
        # just without the position.
        rc_pos, rc_cigar, ref_len = 7, "3M2D2I8M", 30
        _, expected = rc_to_forward_alignment(rc_pos, rc_cigar, ref_len)
        assert rc_cigar_to_forward(rc_pos, rc_cigar, ref_len) == expected

    def test_round_trip_is_identity(self):
        # Flipping the CIGAR twice with consistent positions reproduces it.
        rc_pos, rc_cigar, ref_len = 7, "3M2D2I8M", 30
        fwd_pos, fwd_cigar = rc_to_forward_alignment(rc_pos, rc_cigar, ref_len)
        assert rc_cigar_to_forward(fwd_pos, fwd_cigar, ref_len) == rc_cigar


class TestStateToGlyph:
    """state_to_glyph — render a P/T/M/D state as length + score glyphs."""

    def test_pass_is_check_equal(self):
        assert state_to_glyph("P") == "✓  ="

    def test_tied_is_cross_equal(self):
        assert state_to_glyph("T") == "✗  ="

    def test_missed_is_cross_less(self):
        assert state_to_glyph("M") == "✗  <"

    def test_dominated_is_cross_greater(self):
        assert state_to_glyph("D") == "✗  >"

    def test_unknown_state_raises(self):
        with pytest.raises(ValueError, match="must be one of"):
            state_to_glyph("X")


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


# ---------------------------------------------------------------------------
# sweep / pivot_for_heatmap
# ---------------------------------------------------------------------------

from types import SimpleNamespace

import pandas as pd


class _MockHap:
    """Minimal duck-type stand-in for Haplotype with just body_len."""
    def __init__(self, body_len):
        self.body_len = body_len


def _mock_read(lflank_extent, tag):
    """Read-shaped namespace with a tag attribute so methods can echo it."""
    return SimpleNamespace(
        sequence="A" * 10, var_start=0,
        lflank_extent=lflank_extent, rflank_extent=10,
        tag=tag,
    )


def _make_mock_method(name, fwd_state, rc_state, *, recorder=None):
    """Build a SimpleNamespace method that records its inputs and emits
    canned states.  ``recorder`` is an optional list to append events to."""
    def run(reads, orient):
        if recorder is not None:
            recorder.append((name, "run", orient, len(reads),
                             tuple(r.tag for r in reads)))
        # Return one hit per read; hit is the read itself for traceability.
        return list(reads)

    def truth(r, hap):
        return 100.0

    def classify(hit, r, truth_score, truth_z_bp, orient):
        return fwd_state if orient == "fwd" else rc_state

    return SimpleNamespace(name=name, run=run, truth=truth, classify=classify)


class TestSweep:
    """sweep() — flatten reads, batched run per (method, orient), per-cell classify."""

    def test_returns_long_form_one_row_per_orient(self):
        v = SweepVariant(
            label={"delta": 0}, hap=_MockHap(body_len=15),
            reads=[_mock_read(1, "a"), _mock_read(2, "b")],
        )
        methods = [_make_mock_method("X", "P", "T")]
        df = sweep([v], methods)
        # 2 reads × 1 method × 2 orients = 4 rows
        assert len(df) == 4
        assert set(df.columns) == {"delta", "lflank", "rflank",
                                   "method", "orient", "state"}
        assert sorted(df["orient"].unique()) == ["fwd", "rc"]

    def test_label_columns_propagate(self):
        v1 = SweepVariant(label={"delta": -1, "tag": "x"},
                          hap=_MockHap(10), reads=[_mock_read(1, "a")])
        v2 = SweepVariant(label={"delta":  2, "tag": "y"},
                          hap=_MockHap(20), reads=[_mock_read(3, "b")])
        df = sweep([v1, v2], [_make_mock_method("M", "P", "P")])
        # Each row should carry its variant's label fields.
        v1_rows = df[df["delta"] == -1]
        v2_rows = df[df["delta"] ==  2]
        assert (v1_rows["tag"] == "x").all()
        assert (v2_rows["tag"] == "y").all()

    def test_lflank_comes_from_read(self):
        v = SweepVariant(
            label={}, hap=_MockHap(10),
            reads=[_mock_read(3, "a"), _mock_read(7, "b")],
        )
        df = sweep([v], [_make_mock_method("M", "P", "P")])
        # Each read appears in both orients, so each lflank appears twice.
        assert sorted(df["lflank"].tolist()) == [3, 3, 7, 7]

    def test_rflank_comes_from_read(self):
        # _mock_read hardcodes rflank_extent=10 regardless of lflank, so
        # every emitted row should carry rflank=10.
        v = SweepVariant(
            label={}, hap=_MockHap(10),
            reads=[_mock_read(3, "a"), _mock_read(7, "b")],
        )
        df = sweep([v], [_make_mock_method("M", "P", "P")])
        assert df["rflank"].tolist() == [10, 10, 10, 10]

    def test_state_comes_from_classify(self):
        v = SweepVariant(label={"delta": 0}, hap=_MockHap(10),
                         reads=[_mock_read(1, "a")])
        m = _make_mock_method("M", fwd_state="T", rc_state="D")
        df = sweep([v], [m])
        assert df.set_index("orient")["state"].to_dict() == {"fwd": "T", "rc": "D"}

    def test_run_is_batched_once_per_method_orient(self):
        # Two variants, each contributing 2 reads — sweep must call
        # run() exactly twice per method (once for fwd, once for rc),
        # over all 4 reads at once.
        v1 = SweepVariant(label={"delta": 0}, hap=_MockHap(10),
                          reads=[_mock_read(1, "a"), _mock_read(2, "b")])
        v2 = SweepVariant(label={"delta": 1}, hap=_MockHap(10),
                          reads=[_mock_read(3, "c"), _mock_read(4, "d")])
        events = []
        methods = [
            _make_mock_method("X", "P", "P", recorder=events),
            _make_mock_method("Y", "P", "P", recorder=events),
        ]
        sweep([v1, v2], methods)
        # 2 methods × 2 orients = 4 run() calls, each over all 4 reads.
        runs = [e for e in events if e[1] == "run"]
        assert len(runs) == 4
        for (name, _kind, orient, n_reads, tags) in runs:
            assert n_reads == 4
            assert tags == ("a", "b", "c", "d")
        # And every (method, orient) pair was hit exactly once.
        assert {(name, orient) for (name, _k, orient, *_rest) in runs} == {
            ("X", "fwd"), ("X", "rc"), ("Y", "fwd"), ("Y", "rc"),
        }

    def test_empty_variants_returns_empty_df(self):
        df = sweep([], [_make_mock_method("M", "P", "P")])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# BWACompoundMethod / NWFlexCompoundMethod / wrap_methods_for_multizone_truth
# ---------------------------------------------------------------------------
#
# These tests exercise the classify-side wiring of the compound sweep
# methods with synthetic hits (no real BWA / DP run).  The construction
# tier (Tier 1) and truth-CIGAR tier (Tier 2) covered the building
# blocks; here we verify the sweep harness plumbs them together
# correctly.


class TestBWACompoundMethod:
    """BWACompoundMethod.classify — synthetic-hit wiring tests.

    The constructor only assigns fields; align_bwa runs only inside
    run(), so classify can be exercised with no bwa on PATH.  The
    function picks zones / reference per orient, rescores the hit via
    score_alignment, and forwards to alignment_state_multi under the
    bwa convention.
    """

    SCORE_KW = TestBwaCompoundTruthCigar.SCORE_KW

    @staticmethod
    def _locus():
        return CompoundLocus(
            A="AAAA", R1="AC", N1=3,
            M="TT", R2="GT", N2=2, B="GGGG",
        )

    @classmethod
    def _method(cls):
        loc = cls._locus()
        rc_X, _, _, rc_zones, _ = build_compound_mirror_frame(loc, reads=[])
        m = BWACompoundMethod(
            "BWA-compound", loc, rc_X, rc_zones,
            no_clip=False, score_kwargs=cls.SCORE_KW,
        )
        return m, loc

    def test_truth_cigar_in_fwd_orient_classifies_as_pass(self):
        m, loc = self._method()
        hap = build_compound_haplotype(loc, delta1=1, delta2=-1)
        read = Read(sequence=hap.sequence, var_start=0,
                    lflank_extent=4, rflank_extent=4)
        pos, cigar = bwa_compound_truth_cigar(read, hap, loc)
        truth_score = score_alignment(read.sequence, loc.X, pos, cigar,
                                      **self.SCORE_KW)
        # No .score attribute on the hit — classify must rescore via
        # score_alignment.  An AttributeError here would prove it did not.
        hit = SimpleNamespace(cigar=cigar, pos=pos)
        state = m.classify(hit, read, truth_score, hap.body_lens, "fwd")
        assert state == "P"

    def test_unmapped_hit_classifies_as_dominated(self):
        m, loc = self._method()
        hap = build_compound_haplotype(loc, delta1=0, delta2=0)
        read = Read(sequence=hap.sequence, var_start=0,
                    lflank_extent=4, rflank_extent=4)
        hit = SimpleNamespace(cigar=None, pos=None)
        state = m.classify(hit, read, truth_score=99.0,
                           truth_z_bps=hap.body_lens, orient="fwd")
        assert state == "D"

    def test_length_wrong_hit_uses_rescored_chosen_for_state(self):
        # 10M at pos 11 puts the alignment inside the locus and skips
        # the left flank entirely, so the outer span check fails and
        # the state falls to the score axis.  classify rescores via
        # score_alignment and compares the result to the supplied
        # truth_score to pick T / M / D.
        m, loc = self._method()
        hap = build_compound_haplotype(loc, delta1=0, delta2=0)
        read = Read(sequence=hap.sequence, var_start=0,
                    lflank_extent=4, rflank_extent=4)
        hit = SimpleNamespace(cigar="10M", pos=11)
        chosen = score_alignment(
            read.sequence, loc.X, 11, "10M", **self.SCORE_KW,
        )
        # Same chosen, three truth_scores → T / M / D.
        for truth_score, expected in [
            (chosen,     "T"),
            (chosen + 1, "M"),
            (chosen - 1, "D"),
        ]:
            state = m.classify(hit, read, truth_score=truth_score,
                               truth_z_bps=hap.body_lens, orient="fwd")
            assert state == expected, (truth_score, expected, state)

    def test_rc_orient_rescores_against_rc_reference(self):
        # delta=0 makes hap.sequence == X, so rc(hap) == rc_X.  A 20M
        # hit at pos 1 in the rc frame matches every base and is length-
        # correct against rc_zones (per-block z_bp = body_lens).
        # Demonstrates that the rc branch rescores against rc_X (not X)
        # and dispatches rc_zones (not the fwd zones).
        m, loc = self._method()
        hap = build_compound_haplotype(loc, delta1=0, delta2=0)
        read = Read(sequence=hap.sequence, var_start=0,
                    lflank_extent=4, rflank_extent=4)
        hit = SimpleNamespace(cigar=f"{len(loc.X)}M", pos=1)
        rc_chosen = score_alignment(
            reverse_complement(read.sequence), m.rc_X, 1, hit.cigar,
            **self.SCORE_KW,
        )
        state = m.classify(hit, read, truth_score=rc_chosen,
                           truth_z_bps=hap.body_lens, orient="rc")
        assert state == "P"


class TestNWFlexCompoundMethod:
    """NWFlexCompoundMethod.classify — synthetic-hit wiring tests.

    NW-flex emits NW scores by construction, so classify uses
    ``hit.score`` directly without calling score_alignment.  ep_fwd /
    ep_rc only matter for run(), so we pass None.
    """

    SCORE_KW = TestBwaCompoundTruthCigar.SCORE_KW

    @staticmethod
    def _locus():
        return CompoundLocus(
            A="AAAA", R1="AC", N1=3,
            M="TT", R2="GT", N2=2, B="GGGG",
            nwflex_factor=3,
        )

    @classmethod
    def _method(cls):
        loc = cls._locus()
        _, rc_X_ext, _, _, rc_zones_ext = build_compound_mirror_frame(
            loc, reads=[],
        )
        m = NWFlexCompoundMethod(
            "NWFlex-compound", loc, rc_X_ext, rc_zones_ext,
            ep_fwd=None, ep_rc=None, score_kwargs=cls.SCORE_KW,
        )
        return m, loc

    def test_truth_cigar_in_fwd_orient_classifies_as_pass(self):
        m, loc = self._method()
        hap = build_compound_haplotype(loc, delta1=1, delta2=-1)
        read = Read(sequence=hap.sequence, var_start=0,
                    lflank_extent=4, rflank_extent=4)
        pos, cigar = nwflex_compound_truth_cigar(read, hap, loc)
        truth_score = score_alignment(read.sequence, loc.X_ext, pos, cigar,
                                      **self.SCORE_KW)
        hit = SimpleNamespace(cigar=cigar, pos=pos, score=truth_score)
        state = m.classify(hit, read, truth_score, hap.body_lens, "fwd")
        assert state == "P"

    def test_unmapped_hit_classifies_as_dominated(self):
        m, loc = self._method()
        hap = build_compound_haplotype(loc, delta1=0, delta2=0)
        read = Read(sequence=hap.sequence, var_start=0,
                    lflank_extent=4, rflank_extent=4)
        # alignment_state_multi short-circuits to D when cigar is None;
        # hit.score still needs to be float-convertible because classify
        # passes it through float() unconditionally.
        hit = SimpleNamespace(cigar=None, pos=None, score=0.0)
        state = m.classify(hit, read, truth_score=99.0,
                           truth_z_bps=hap.body_lens, orient="fwd")
        assert state == "D"

    def test_hit_score_is_used_directly_without_rescore(self):
        # Lie about hit.score to prove classify did not silently rescore.
        # 10M at pos 11 is length-wrong (no flank left).  If classify had
        # rescored via score_alignment the chosen would be a strongly
        # negative value (mismatches), and state at truth_score=500
        # would be M.  Setting hit.score=500 instead, the state must be
        # T (chosen == truth) — only reachable if hit.score was used.
        m, loc = self._method()
        hap = build_compound_haplotype(loc, delta1=0, delta2=0)
        read = Read(sequence=hap.sequence, var_start=0,
                    lflank_extent=4, rflank_extent=4)
        hit = SimpleNamespace(cigar="10M", pos=11, score=500.0)
        state = m.classify(hit, read, truth_score=500.0,
                           truth_z_bps=hap.body_lens, orient="fwd")
        assert state == "T"


class TestWrapMethodsForMultizoneTruth:
    """wrap_methods_for_multizone_truth — rewires sweep's scalar
    truth_z_bp into the haplotype's per-block ``body_lens`` tuple by
    looking the read up by ``id()`` in the variants list.  name, run,
    truth pass through; only classify is rebound.
    """

    @staticmethod
    def _hap(body_lens):
        # SimpleNamespace is fine — wrap only reads .body_lens, and the
        # caller-side body_len scalar we pretend sweep would pass.
        return SimpleNamespace(body_lens=body_lens,
                               body_len=sum(body_lens))

    @staticmethod
    def _read(tag):
        return SimpleNamespace(sequence=tag, var_start=0,
                               lflank_extent=1, rflank_extent=1)

    def test_name_run_and_truth_pass_through_unchanged(self):
        r = self._read("a")
        v = SweepVariant(label={}, hap=self._hap((3, 5)), reads=[r])
        base_run = lambda reads, orient: ["x"]
        base_truth = lambda read, hap: 42.0
        m = SimpleNamespace(
            name="m-name", run=base_run, truth=base_truth,
            classify=lambda *a, **k: "P",
        )
        (wrapped,) = wrap_methods_for_multizone_truth([m], [v])
        assert wrapped.name == "m-name"
        assert wrapped.run is base_run
        assert wrapped.truth is base_truth

    def test_classify_substitutes_body_lens_for_scalar_truth_z_bp(self):
        # Sweep would thread v.hap.body_len (a scalar) through every
        # classify() call.  The wrapper must replace that scalar with
        # the haplotype's body_lens tuple so the multi-zone classifier
        # downstream gets per-block lengths.
        r = self._read("a")
        h = self._hap((7, 11))
        v = SweepVariant(label={}, hap=h, reads=[r])
        seen = []
        def base_classify(hit, r, truth_score, truth_z_bp_passed, orient):
            seen.append(truth_z_bp_passed)
            return "P"
        m = SimpleNamespace(
            name="m", run=lambda *a, **k: [], truth=lambda *a, **k: 0.0,
            classify=base_classify,
        )
        (wrapped,) = wrap_methods_for_multizone_truth([m], [v])
        wrapped.classify(None, r, 0.0, h.body_len, "fwd")
        # Whatever scalar we passed in, the base classify receives the tuple.
        assert seen == [(7, 11)]

    def test_classify_resolves_reads_to_their_owning_variants_hap(self):
        # Two variants, each with its own read and its own body_lens.
        # The wrapper looks reads up by id(), so cross-variant routing
        # must keep each read pointed at its own haplotype.
        r_a, r_b = self._read("a"), self._read("b")
        h_a, h_b = self._hap((2, 4)), self._hap((10, 20))
        v_a = SweepVariant(label={}, hap=h_a, reads=[r_a])
        v_b = SweepVariant(label={}, hap=h_b, reads=[r_b])
        seen = []
        def base_classify(hit, r, truth_score, truth_z_bp_passed, orient):
            seen.append((r.sequence, truth_z_bp_passed))
            return "P"
        m = SimpleNamespace(
            name="m", run=lambda *a, **k: [], truth=lambda *a, **k: 0.0,
            classify=base_classify,
        )
        (wrapped,) = wrap_methods_for_multizone_truth([m], [v_a, v_b])
        wrapped.classify(None, r_a, 0.0, h_a.body_len, "fwd")
        wrapped.classify(None, r_b, 0.0, h_b.body_len, "fwd")
        assert seen == [("a", (2, 4)), ("b", (10, 20))]


class TestPivotForHeatmap:
    """pivot_for_heatmap() — long-form -> arm-wide for the heatmap helper."""

    def _long(self, rows):
        return pd.DataFrame(rows)

    def test_pivots_orient_into_columns(self):
        long = self._long([
            {"delta": 0, "lflank": 1, "method": "X", "orient": "fwd", "state": "P"},
            {"delta": 0, "lflank": 1, "method": "X", "orient": "rc",  "state": "P"},
        ])
        wide = pivot_for_heatmap(long)
        assert set(wide.columns) >= {"delta", "lflank", "arm",
                                     "fwd_state", "rc_state", "state"}
        row = wide.iloc[0]
        assert row["fwd_state"] == "P"
        assert row["rc_state"]  == "P"

    def test_renames_method_to_arm(self):
        long = self._long([
            {"delta": 0, "lflank": 1, "method": "BWA-std", "orient": "fwd", "state": "P"},
            {"delta": 0, "lflank": 1, "method": "BWA-std", "orient": "rc",  "state": "P"},
        ])
        wide = pivot_for_heatmap(long)
        assert "method" not in wide.columns
        assert wide["arm"].tolist() == ["BWA-std"]

    def test_combined_state_best_policy(self):
        # combine_states("T", "D", "best") = "T" (T has lower priority number).
        long = self._long([
            {"delta": 0, "lflank": 1, "method": "X", "orient": "fwd", "state": "T"},
            {"delta": 0, "lflank": 1, "method": "X", "orient": "rc",  "state": "D"},
        ])
        wide = pivot_for_heatmap(long, combine="best")
        assert wide["state"].tolist() == ["T"]

    def test_combined_state_worst_policy(self):
        long = self._long([
            {"delta": 0, "lflank": 1, "method": "X", "orient": "fwd", "state": "T"},
            {"delta": 0, "lflank": 1, "method": "X", "orient": "rc",  "state": "D"},
        ])
        wide = pivot_for_heatmap(long, combine="worst")
        assert wide["state"].tolist() == ["D"]

    def test_one_row_per_arm_per_cell(self):
        long = self._long([
            {"delta": 0, "lflank": 1, "method": "X", "orient": "fwd", "state": "P"},
            {"delta": 0, "lflank": 1, "method": "X", "orient": "rc",  "state": "P"},
            {"delta": 0, "lflank": 1, "method": "Y", "orient": "fwd", "state": "D"},
            {"delta": 0, "lflank": 1, "method": "Y", "orient": "rc",  "state": "D"},
            {"delta": 1, "lflank": 1, "method": "X", "orient": "fwd", "state": "T"},
            {"delta": 1, "lflank": 1, "method": "X", "orient": "rc",  "state": "T"},
            {"delta": 1, "lflank": 1, "method": "Y", "orient": "fwd", "state": "M"},
            {"delta": 1, "lflank": 1, "method": "Y", "orient": "rc",  "state": "M"},
        ])
        wide = pivot_for_heatmap(long)
        # 2 deltas × 1 lflank × 2 arms = 4 rows
        assert len(wide) == 4
        assert sorted(wide["arm"].unique().tolist()) == ["X", "Y"]


class TestPlotCorrectnessHeatmapRows:
    """plot_correctness_heatmap_rows — multi-row stacked heatmap grid."""

    @staticmethod
    def _df(deltas, lflanks, arms):
        return pd.DataFrame([
            {"delta": d, "lflank": L, "arm": a,
             "fwd_state": "P", "rc_state": "T", "state": "P"}
            for d in deltas for L in lflanks for a in arms
        ])

    def test_axis_count_equals_rows_times_arms(self):
        import matplotlib
        matplotlib.use("Agg")
        deltas, lflanks = [-1, 0, 1], [1, 2]
        arms = {"X": "X-arm", "Y": "Y-arm", "Z": "Z-arm"}
        df = self._df(deltas, lflanks, arms)
        rows = [(k, df) for k in (10, 20, 30, 40)]
        fig = plot_correctness_heatmap_rows(
            rows, deltas=deltas, lflanks=lflanks, arm_titles=arms,
            row_label_fn=lambda k: f"k={k}",
        )
        # 4 rows × 3 arms = 12 panels.  Legend doesn't add an Axes
        # (it's a fig.legend, not an inset axes).
        assert len(fig.axes) == 4 * 3

    def test_column_titles_only_on_top_row(self):
        import matplotlib
        matplotlib.use("Agg")
        deltas, lflanks = [-1, 0, 1], [1, 2]
        arms = {"X": "X-arm", "Y": "Y-arm"}
        df = self._df(deltas, lflanks, arms)
        rows = [(k, df) for k in (1, 2)]
        fig = plot_correctness_heatmap_rows(
            rows, deltas=deltas, lflanks=lflanks, arm_titles=arms,
            row_label_fn=lambda k: f"k={k}",
        )
        # axes ordering from plt.subplots(n_rows, n_cols) is row-major.
        top_row = fig.axes[:2]
        bottom_row = fig.axes[2:4]
        assert [ax.get_title() for ax in top_row] == ["X-arm", "Y-arm"]
        assert [ax.get_title() for ax in bottom_row] == ["", ""]

    def test_row_labels_use_row_label_fn(self):
        import matplotlib
        matplotlib.use("Agg")
        deltas, lflanks = [-1, 0, 1], [1, 2]
        arms = {"X": "X-arm"}
        df = self._df(deltas, lflanks, arms)
        rows = [("alpha", df), ("beta", df)]
        fig = plot_correctness_heatmap_rows(
            rows, deltas=deltas, lflanks=lflanks, arm_titles=arms,
            row_label_fn=lambda k: f"row {k}",
        )
        # First column of each row carries the row label.
        first_col_ylabels = [fig.axes[0].get_ylabel(), fig.axes[1].get_ylabel()]
        assert first_col_ylabels[0].startswith("row alpha")
        assert first_col_ylabels[1].startswith("row beta")
