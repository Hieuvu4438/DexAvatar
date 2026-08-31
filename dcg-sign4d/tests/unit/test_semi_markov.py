import itertools

import torch

from dcg_sign4d.contact.ontology import VALID_FRAME_TRANSITIONS
from dcg_sign4d.contact.semi_markov import SemiMarkovDecoder


def is_valid(sequence, max_duration):
    run = 1
    for index in range(1, len(sequence)):
        if not VALID_FRAME_TRANSITIONS[sequence[index - 1], sequence[index]]:
            return False
        run = run + 1 if sequence[index] == sequence[index - 1] else 1
        if sequence[index] in (1, 3) and run > 1:
            return False
        if run > max_duration:
            return False
    return sequence[0] not in (1, 3) or len(sequence) == 1 or sequence[1] != sequence[0]


def path_score(sequence, logits, durations):
    total = 0.0
    start = 0
    for end in range(1, len(sequence) + 1):
        if end == len(sequence) or sequence[end] != sequence[start]:
            state = sequence[start]
            total += float(logits[start:end, state].sum())
            total += float(durations[start, end - start - 1])
            start = end
    return total


def test_exact_decoder_matches_brute_force():
    torch.manual_seed(4)
    logits = torch.randn(1, 4, 1, 4)
    durations = torch.randn(1, 4, 1, 4)
    graph = SemiMarkovDecoder(4).decode(logits, durations, torch.tensor([[True]]))
    decoded = tuple(graph.event_state[0, :, 0].tolist())
    candidates = [x for x in itertools.product(range(4), repeat=4) if is_valid(x, 4)]
    best = max(candidates, key=lambda x: path_score(x, logits[0, :, 0], durations[0, :, 0]))
    assert decoded == best


def test_known_chain_padding_and_determinism():
    logits = torch.full((1, 6, 1, 4), -20.0)
    target = torch.tensor([0, 1, 2, 2, 3, 0])
    logits[0, torch.arange(6), 0, target] = 20.0
    durations = torch.zeros(1, 6, 1, 6)
    decoder = SemiMarkovDecoder(6)
    valid = torch.tensor([[True, True, True, True, True, False]])
    first = decoder.decode(logits, durations, torch.tensor([[True]]), valid)
    second = decoder.decode(logits, durations, torch.tensor([[True]]), valid)
    assert first.event_state[0, :5, 0].tolist() == [0, 1, 2, 2, 3]
    assert first.segment_id[0, 5, 0] == -1
    assert torch.equal(first.event_state, second.event_state)


def test_invalid_transition_never_appears():
    logits = torch.zeros(1, 7, 2, 4)
    durations = torch.zeros(1, 7, 2, 7)
    graph = SemiMarkovDecoder(7).decode(logits, durations, torch.ones(1, 2, dtype=torch.bool))
    previous = graph.event_state[:, :-1]
    current = graph.event_state[:, 1:]
    assert bool(VALID_FRAME_TRANSITIONS[previous, current].all())
