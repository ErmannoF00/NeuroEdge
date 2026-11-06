"""CTC symbol IDs match the published Brain-to-Text '25 baseline.

Source for mapping: Neuroprosthetics-Lab/nejm-brain-to-text,
model_training/evaluate_model_helpers.py, LOGIT_TO_PHONEME.
0 is CTC blank; indices 1..40 are phonemes and word boundary.
"""
from __future__ import annotations

TOKENS = (
    "BLANK", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L",
    "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH",
    "UH", "UW", "V", "W", "Y", "Z", "ZH", " | ",
)
BLANK = 0
NUM_CLASSES = len(TOKENS)


def ctc_collapse(path: list[int], blank: int = BLANK) -> list[int]:
    """Collapse consecutive *frame* repeats first, then remove blank.

    [A, A, blank, A] -> [A,A], unlike removing blanks first.
    """
    result: list[int] = []
    previous = None
    for idx in path:
        idx = int(idx)
        if idx != previous and idx != blank:
            result.append(idx)
        previous = idx
    return result


def edit_distance(a: list[int], b: list[int]) -> int:
    """Levenshtein distance in O(min(len(a),len(b))) memory."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j-1] + (x != y)))
        prev = cur
    return prev[-1]


def ctc_min_length(target: list[int]) -> int:
    """CTC requires extra blank frames between *consecutive identical* labels."""
    return len(target) + sum(x == y for x, y in zip(target, target[1:]))
