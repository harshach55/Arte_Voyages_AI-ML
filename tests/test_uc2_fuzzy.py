import pytest

from src.uc2_extractor.fuzzy import close_match, edit_distance, word_spans


@pytest.mark.parametrize(
    "a, b, d",
    [("novmber", "november", 1), ("febuary", "february", 1), ("maledieven", "malediven", 1),
     ("sri lnka", "sri lanka", 1), ("rajsathan", "rajasthan", 1), ("kerala", "kerala", 0),
     ("paris", "nepal", 2)],
)
def test_edit_distance(a, b, d):
    assert min(edit_distance(a, b, limit=1), 2) == min(d, 2)


def test_close_match_needs_same_first_letter_and_min_length():
    forms = ["november", "nepal", "kerala"]
    assert close_match("Novmber", forms) == "november"
    assert close_match("Kerela", forms) == "kerala"
    assert close_match("Nepl", forms) is None  # "nepal" is too short for fuzzy matching
    assert close_match("Movember", forms) is None  # different first letter


def test_close_match_ignores_exact_and_unclear_words():
    assert close_match("kerala", ["kerala"]) is None
    assert close_match("abcdef", ["abcdeg", "abcdeh"]) is None


def test_word_spans_keep_offsets():
    text = "nach Sri Lnka, bitte"
    for s, e, t in word_spans(text):
        assert text[s:e] == t
    assert (5, 13, "Sri Lnka") in word_spans(text)
