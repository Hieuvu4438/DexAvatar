from signpk.data.window_sampler import make_window


def test_reflect_window_is_bidirectional_and_logged():
    window = make_window(center=0, length=5, window=5, gap=1, padding="reflect")
    assert window.indices == (2, 1, 0, 1, 2)
    assert window.padded == (True, True, False, False, False)
    assert window.padding_ratio == 0.4


def test_replicate_window():
    assert make_window(4, 5, 5, 1, "replicate").indices == (2, 3, 4, 4, 4)

