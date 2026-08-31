from cusp_sl.evaluate_author import METRIC_PATTERN


def test_author_metric_parser_ignores_per_sign_lines():
    text = """
Right hand:  10.0
[method_hamer]: V2V Left Wrist: 13.5735 (mm)
[method_hamer]: Tr Above Pelvis Minus Face: 29.9074 (mm)
"""
    assert METRIC_PATTERN.findall(text) == [
        ("V2V Left Wrist", "13.5735"),
        ("Tr Above Pelvis Minus Face", "29.9074"),
    ]
