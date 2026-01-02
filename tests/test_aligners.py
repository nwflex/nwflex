# test_aligners.py

import pytest

from nwflex.aligners import (
    align_single_block,
    alignment_to_cigar,
    _parse_cigar,
    _write_cigar,
    _op_length_total,
    rle_ops,
    align_standard,
)
from nwflex import default

class TestCigarParsing:
    """ Unit tests for CIGAR string parsing. """
    def test_parse_simple(self):
        assert _parse_cigar("4M") == [(4, "M")]
        assert _parse_cigar("10M5I3D") == [(10, "M"), (5, "I"), (3, "D")]
        assert _parse_cigar("") == []

    def test_parse_invalid(self):
        pytest.raises(ValueError, _parse_cigar, "4Z")            # Invalid operation
        pytest.raises(ValueError, _parse_cigar, "M4")            # Invalid format
        pytest.raises(ValueError, _parse_cigar, "4")             # Missing operation
        pytest.raises(ValueError, _parse_cigar, "4Mhello10M")    # invalid interrupting text

class TestRleOps:
    """ Unit tests for run-length encoding operations. """
    def test_rle_ops(self):
        assert rle_ops("MMMM") == [(4, "M")]
        assert rle_ops("MMIIIDDDMM") == [(2, "M"), (3, "I"), (3, "D"), (2, "M")]
        assert rle_ops("") == []

class TestAlignmentToCigar:
    """ Integration tests: alignment to CIGAR string conversion. """
    def test_perfect_match(self):
        X = "ACGT"
        Y = "ACGT"
        result = align_standard(X, Y, **default.align_params())
        start_pos, cigar = alignment_to_cigar(result.path)
        assert (start_pos, cigar) == (1, "4M")

    def test_insertion(self):
        X = "ACGT"
        Y = "ACTTGT"
        result = align_standard(X, Y, **default.align_params())
        start_pos, cigar = alignment_to_cigar(result.path)
        assert (start_pos, cigar) == (1, "2M2I2M")

    def test_deletion(self):        
        X = "ACTTGT"
        Y = "ACGT"
        result = align_standard(X, Y, **default.align_params())
        start_pos, cigar = alignment_to_cigar(result.path)
        assert (start_pos, cigar) == (1, "2M2D2M")
    
    def test_softclip(self):
        """ softclipped bases on both sides from semiglobal alignment """
        X = "ACGT"
        Y = "TTACGTTT"
        result = align_standard(X, Y, **default.align_params(semiglobal=True))
        start_pos, cigar = alignment_to_cigar(result.path)
        assert (start_pos, cigar) == (1, "2S4M2S")

    def test_semiglobal(self):
        """ shifted start position from semiglobal alignment """
        X = "TTTTACGT"
        Y = "ACGT"
        result = align_standard(X, Y, **default.align_params(semiglobal=True))
        start_pos, cigar = alignment_to_cigar(result.path)
        assert (start_pos, cigar) == (5, "4M")

    def test_nwflex(self):
        X = "ACTTTTGT"
        Y = "ACTTGT"
        result = align_single_block(X, Y, s=2, e=6, **default.align_params())
        start_pos, cigar = alignment_to_cigar(result.path)
        assert (start_pos, cigar) == (1, "2M2N4M")

    def test_mismatch(self):
        X = "ACGT"
        Y = "AGGT"
        result = align_standard(X, Y, **default.align_params())
        start_pos, cigar = alignment_to_cigar(result.path)
        assert (start_pos, cigar) == (1, "4M")
