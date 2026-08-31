from cusp_sl.bind_generator_checkpoint import bind_payload


def test_bind_payload_preserves_model_and_records_provenance():
    model = {"weight": object()}
    source = {"model": model, "step": 10}
    bound = bind_payload(source, source_sha256="a" * 64, q_sha256="b" * 64)
    assert bound["model"] is model
    assert bound["step"] == 10
    assert bound["source_checkpoint_sha256"] == "a" * 64
    assert bound["reliability_checkpoint_sha256"] == "b" * 64
    assert "reliability_checkpoint_sha256" not in source
