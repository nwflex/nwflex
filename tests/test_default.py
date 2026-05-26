"""
test_default.py — Tests for the scoring preset registry.

Verifies that named presets resolve correctly, that helpers honor the
`preset` parameter, and that the module-level constants stay in lockstep
with `PRESETS["nwflex_default"]`.
"""

import pytest

from nwflex import default
from nwflex.default import (
    PRESETS,
    ScoringPreset,
    align_params,
    get_default_scoring,
    get_preset,
    make_dna_score_matrix,
)


class TestMakeDnaScoreMatrix:
    def test_diagonal_and_offdiagonal(self):
        M = make_dna_score_matrix(1.0, -4.0)
        assert M.shape == (4, 4)
        assert (M.diagonal() == 1.0).all()
        assert M[0, 1] == -4.0 and M[2, 3] == -4.0 and M[3, 0] == -4.0


class TestPresets:
    def test_known_presets_present(self):
        assert "nwflex_default" in PRESETS
        assert "bwa_mem" in PRESETS

    def test_get_preset_returns_scoring_preset(self):
        p = get_preset("bwa_mem")
        assert isinstance(p, ScoringPreset)
        assert p.name == "bwa_mem"

    def test_bwa_mem_values(self):
        p = get_preset("bwa_mem")
        assert p.score_matrix[0, 0] == 1.0
        assert p.score_matrix[0, 1] == -4.0
        assert p.gap_open == -6.0
        assert p.gap_extend == -1.0

    def test_unknown_preset_raises_keyerror_listing_options(self):
        with pytest.raises(KeyError) as exc:
            get_preset("does_not_exist")
        msg = str(exc.value)
        assert "nwflex_default" in msg and "bwa_mem" in msg


class TestHelpers:
    def test_get_default_scoring_default_matches_constants(self):
        sm, go, ge, a2i = get_default_scoring()
        assert sm is default.SCORE_MATRIX
        assert go == default.GAP_OPEN
        assert ge == default.GAP_EXTEND
        assert a2i is default.ALPHABET_TO_INDEX

    def test_get_default_scoring_with_preset(self):
        sm, go, ge, _ = get_default_scoring("bwa_mem")
        assert sm[0, 0] == 1.0 and sm[0, 1] == -4.0
        assert go == -6.0 and ge == -1.0

    def test_align_params_with_preset(self):
        kwargs = align_params(preset="bwa_mem", semiglobal=True)
        assert kwargs["gap_open"] == -6.0
        assert kwargs["gap_extend"] == -1.0
        assert kwargs["free_X"] is True and kwargs["free_Y"] is True

    def test_module_constants_alias_default_preset(self):
        # Single source of truth: constants and the default preset must agree.
        p = PRESETS["nwflex_default"]
        assert default.SCORE_MATRIX is p.score_matrix
        assert default.GAP_OPEN == p.gap_open
        assert default.GAP_EXTEND == p.gap_extend
