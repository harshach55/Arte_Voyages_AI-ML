"""Tests for the info icons on both demo pages."""

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "app" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


demo = load("uc2_web_demo")
highlight = load("uc2_demo_highlight")
info = load("uc2_info")
PAGE_8000 = demo.render_page(demo.SAMPLES, False, "x").decode("utf-8")
PAGE_8002 = highlight.render_page(highlight.sample_list(), "Mode").decode("utf-8")


def slots(page):
    return set(re.findall(r'data-info-slot="(\w+)"', page.split("<script>")[0]))


def test_every_icon_has_wording():
    assert slots(PAGE_8000) <= set(info.INFO) and slots(PAGE_8002) <= set(info.INFO)


def test_main_page_has_its_icons():
    assert {"mode", "several", "source", "tag_rule", "tag_model", "tag_crf", "flags", "save", "pipeline"} == slots(PAGE_8000)
    for key in ("real_number", "optimistic", "unclear_dates", "budget_band", "language"):
        assert f'"{key}"' in PAGE_8000


def test_highlight_page_has_only_what_exists_there():
    assert {"mode", "tag_rule", "tag_crf", "highlight"} == slots(PAGE_8002)


def test_both_pages_carry_the_same_wording():
    for text in info.INFO.values():
        assert text in PAGE_8000 and text in PAGE_8002


def test_popups_use_plain_text_only():
    for page in (PAGE_8000, PAGE_8002):
        assert "innerHTML" not in page and "__INFO" not in page


def test_popup_works_on_hover_focus_and_tap():
    for event in ("mouseover", "focusin", "click", "keydown"):
        assert f'addEventListener("{event}"' in info.INFO_JS
