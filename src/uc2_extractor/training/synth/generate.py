"""Generate synthetic training enquiries, check every one, and report what was rejected.

Every example must pass validate_example (each span matches the enquiry text at its offsets,
sits on token boundaries and survives the BIO round trip) and must not be an exact or near
duplicate of any enquiry in the gold set, held-out sets 1 and 2 or the corrections.

Usage:
    python -m src.uc2_extractor.training.synth.generate --per-language 60 --out <file.jsonl>
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Optional

from ..common import TRAINING_DIR, label_counts, reject_reason, write_jsonl
from ..dedupe import ProtectedIndex, build_index
from . import lang_de, lang_en, lang_fr
from .compose import Writer
from .parts import assemble
from .trips import sample_trip

LANGUAGES = {"en": lang_en, "fr": lang_fr, "de": lang_de}
STYLES = (("long", "medium", "short"), (4, 3, 3))
DEFAULT_SEED = 20261106


def make_example(language: str, index: int, rng: random.Random) -> dict:
    style = rng.choices(STYLES[0], weights=STYLES[1])[0]
    trip = sample_trip(language, style, rng)
    parts, meta = Writer(LANGUAGES[language], rng).write(trip)
    text, spans = assemble(parts)
    return {"id": f"synth-{language}-{index:06d}", "source": "synthetic", "language": language,
            "text": text, "spans": spans, "ignore": [], "meta": meta}


def generate(per_language: int, seed: int = DEFAULT_SEED, index: Optional[ProtectedIndex] = None,
             max_attempts: int = 50) -> tuple[list[dict], Counter]:
    """Generate exactly per_language accepted examples per language, unless too many are rejected."""
    index = index or ProtectedIndex()
    kept: list[dict] = []
    dropped: Counter = Counter()
    for language in LANGUAGES:
        rng = random.Random(f"{seed}-{language}")
        seen: set[str] = set()
        accepted = 0
        for attempt in range(per_language * max_attempts):
            if accepted == per_language:
                break
            example = make_example(language, attempt, rng)
            reason = reject_reason(example, index, seen)
            if reason:
                dropped[reason] += 1
            else:
                kept.append(example)
                accepted += 1
    return kept, dropped


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-language", type=int, default=60)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=TRAINING_DIR / "synthetic" / "pilot.jsonl")
    args = parser.parse_args(argv)
    index, protected = build_index()
    examples, dropped = generate(args.per_language, args.seed, index)
    write_jsonl(args.out, examples)
    styles = Counter((e["language"], e["meta"]["style"]) for e in examples)
    report = {"kept": len(examples), "rejected": dict(dropped), "protected_texts": protected,
              "styles": {f"{k[0]}-{k[1]}": v for k, v in sorted(styles.items())},
              "with_typos": sum(1 for e in examples if e["meta"]["typo_rate"]),
              "labels": label_counts(examples)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
