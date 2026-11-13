"""Tests for the highlight page: pieces keep the exact text, and text can never run as code."""

import importlib.util
import json
from pathlib import Path

import pytest

from src.uc2_extractor.schema import Evidence

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("uc2_demo_highlight", ROOT / "app" / "uc2_demo_highlight.py")
page = importlib.util.module_from_spec(spec)
spec.loader.exec_module(page)


def ev(field, text, whole):
    start = whole.index(text)
    return Evidence(field=field, text=text, start=start, end=start + len(text))


def test_pieces_join_back_to_the_exact_text():
    text = "Two weeks in Kerala, budget €9,000.\nVegetarian."
    pieces = page.highlight_segments(text, [ev("destination", "Kerala", text), ev("budget", "€9,000", text)])
    assert "".join(p["text"] for p in pieces) == text
    assert [p["fields"] for p in pieces if p["fields"]] == [["destination"], ["budget"]]


def test_overlapping_evidence_keeps_both_fields():
    text = "beachfront villa"
    pieces = page.highlight_segments(text, [ev("accommodation", "beachfront villa", text), ev("interests", "beachfront", text)])
    assert pieces == [{"text": "beachfront", "fields": ["accommodation", "interests"]},
                      {"text": " villa", "fields": ["accommodation"]}]


def test_evidence_that_does_not_match_the_text_is_skipped():
    bad = Evidence(field="dates", text="May", start=0, end=3)
    assert page.highlight_segments("Goa", [bad]) == [{"text": "Goa", "fields": []}]


def test_no_evidence_gives_the_plain_text():
    assert page.highlight_segments("Hello there", []) == [{"text": "Hello there", "fields": []}]


def test_html_in_an_enquiry_stays_plain_text_in_the_data():
    text = "<script>alert(1)</script> Two weeks in Kerala"
    data = page.build_response(text, False)
    assert "".join(p["text"] for p in data["segments"]) == text


def test_page_never_inserts_html():
    html = page.render_page(page.sample_list(), "Mode").decode("utf-8")
    assert "innerHTML" not in html and "insertAdjacentHTML" not in html and "document.write" not in html


def test_page_data_cannot_close_the_script_block():
    html = page.render_page([{"label": "x", "text": "</script><script>alert(1)</script>"}], "</script>").decode("utf-8")
    assert "</script><script>alert(1)" not in html.split("<script>", 1)[1]


def test_page_has_a_colour_for_every_field_the_values_can_have():
    assert {"destination", "dates", "travellers", "budget", "dietary", "occasion"} <= set(page.COLOURS)
    assert len({colour for _, colour in page.COLOURS.values()}) == len(page.COLOURS)


@pytest.mark.parametrize("item, expected", [
    ({"source": "rule", "confidence": None}, "rule"),
    ({"source": "crf", "confidence": 0.807}, "crf 0.81"),
    ({"source": "rule+crf", "confidence": 0.5}, "rule + crf 0.50"),
])
def test_tag_text(item, expected):
    assert page.tag_text(item) == expected


def test_table_rows_carry_value_field_and_tag():
    rows = page.build_response("We want Kerala, vegetarian food please", False)["rows"]
    json.dumps(rows)
    assert {"value": "Kerala", "field": "Destination", "key": "destination", "tag": "rule"} in rows
    assert {"value": "vegetarian", "field": "Diet", "key": "dietary", "tag": "rule"} in rows
