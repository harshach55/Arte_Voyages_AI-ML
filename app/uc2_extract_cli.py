"""Command-line demo: paste a travel enquiry, get the client profile as JSON.

Examples (from the project root):
    python app/uc2_extract_cli.py --text "My wife and I want 2 weeks in Kerala in March."
    python app/uc2_extract_cli.py --file my_enquiry.txt
    python app/uc2_extract_cli.py
        (then paste the enquiry and finish with Ctrl+Z then Enter on Windows,
         or Ctrl+D on macOS/Linux)

Options:
    --no-evidence   leave out the supporting text spans
    --compact       print the JSON on one line
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.uc2_extractor.extract import extract_profile  # noqa: E402


def read_enquiry(args) -> str:
    if args.text is not None:
        return args.text
    if args.file is not None:
        return Path(args.file).read_text(encoding="utf-8")
    if sys.stdin.isatty():
        print("Paste the enquiry, then press Ctrl+Z and Enter (Windows) or Ctrl+D (macOS/Linux):",
              file=sys.stderr)
    return sys.stdin.read()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Extract a client profile from a travel enquiry (EN/FR/DE).")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="the enquiry text")
    source.add_argument("--file", help="a UTF-8 text file with the enquiry")
    parser.add_argument("--no-evidence", action="store_true", help="leave out the supporting text spans")
    parser.add_argument("--compact", action="store_true", help="print the JSON on one line")
    args = parser.parse_args(argv)

    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    text = read_enquiry(args).strip()
    if not text:
        print("No enquiry text given.", file=sys.stderr)
        return 1

    profile = extract_profile(text)
    data = profile.model_dump(mode="json", exclude={"evidence"} if args.no_evidence else None)
    print(json.dumps(data, ensure_ascii=False, indent=None if args.compact else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
