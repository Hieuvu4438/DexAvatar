from cusp_sl.render_predictions import use_bit_exact_base_copy


def test_renderer_preserves_legacy_identity_but_renders_strong_a1_base():
    assert use_bit_exact_base_copy(0, {"input_manifest_role": "legacy_unspecified"})
    assert not use_bit_exact_base_copy(
        0, {"input_manifest_role": "frozen_strong_a1_derived_cache"}
    )
    assert not use_bit_exact_base_copy(1, {"input_manifest_role": "legacy_unspecified"})
