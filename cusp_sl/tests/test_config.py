from cusp_sl.config import load_config


def test_release_config_is_valid():
    config = load_config("cusp_sl/configs/cusp_sl_v1.yaml")
    assert config.protocol.expected_frames == 1493
    assert config.flow.ode_steps == 3
    assert not config.form.enabled


def test_v2_config_declares_train_only_normalization_and_tighter_q_target():
    config = load_config("cusp_sl/configs/cusp_sl_v2_normalized.yaml")
    assert config.flow.normalization_statistics.endswith("residual_statistics_train.npz")
    assert config.reliability.body_tolerance_degrees == 3.0
    assert config.reliability.hand_tolerance_degrees == 5.0
