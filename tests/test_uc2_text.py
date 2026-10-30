import pytest

from src.uc2_extractor.text import detect_language, tokenize

SAMPLES = [
    "We'd like 2 weeks in Sri Lanka, budget $20,000.",
    "Nous aimerions découvrir l'Inde du 3 au 17 mars, budget 15 000 €.",
    "Wir möchten im März nach Indien reisen – Budget 20.000 Euro, Ankunft 3/4.",
    "  leading spaces,\ttabs\nand new lines  ",
    "Émilie & Jürgen: 5-Sterne-Hotel, à Kochi!",
]


@pytest.mark.parametrize("text", SAMPLES)
def test_every_token_matches_its_offsets(text):
    for tok in tokenize(text):
        assert text[tok.start:tok.end] == tok.text


@pytest.mark.parametrize("text", SAMPLES)
def test_tokens_are_in_order_and_do_not_overlap(text):
    toks = tokenize(text)
    for a, b in zip(toks, toks[1:]):
        assert a.end <= b.start


@pytest.mark.parametrize("text", SAMPLES)
def test_only_whitespace_is_dropped(text):
    covered = set()
    for tok in tokenize(text):
        covered.update(range(tok.start, tok.end))
    for i, ch in enumerate(text):
        assert (i in covered) or ch.isspace()


def test_numbers_with_separators_stay_whole():
    words = [t.text for t in tokenize("20,000 and 20.000 and 3/4 and 15:30")]
    assert words == ["20,000", "and", "20.000", "and", "3/4", "and", "15:30"]


def test_hyphenated_words_stay_whole_and_apostrophes_split():
    words = [t.text for t in tokenize("five-star l'Inde")]
    assert words == ["five-star", "l", "'", "Inde"]


def test_accented_offsets():
    text = "Hôtel à Négombo"
    toks = tokenize(text)
    assert toks[2].text == "Négombo"
    assert (toks[2].start, toks[2].end) == (8, 15)


def test_empty_text():
    assert tokenize("") == []


@pytest.mark.parametrize(
    "text, lang",
    [
        ("We would like to travel to India with our two children in March.", "en"),
        ("Nous voudrions partir en Inde avec nos deux enfants au mois de mars.", "fr"),
        ("Wir möchten mit unseren zwei Kindern im März nach Indien reisen.", "de"),
    ],
)
def test_detect_language(text, lang):
    assert detect_language(text) == lang


@pytest.mark.parametrize("text", ["", "Sri Lanka 2027", "Kerala, Goa, Delhi"])
def test_detect_language_returns_none_without_signal(text):
    assert detect_language(text) is None
