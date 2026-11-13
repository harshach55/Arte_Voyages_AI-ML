"""Train the CRF slot tagger on the converted public data, and report it on a development split.

Data: data/uc2_inquiries/training/converted/massive.jsonl and taskmaster2.jsonl.

Development split, taken from the same public sources:
- MASSIVE: its own partitions. train is used for training, dev and test for development. A
  sentence and its translations share one partition, so a translation cannot cross the split.
- Taskmaster-2: whole dialogues, chosen by a hash of the dialogue id (DEV_SHARE percent).

Text that a source annotated with something we do not use ("ignore" in the data) is left out
of training: each sentence is cut at those tokens, so the tagger is not told they are plain
text. On the development split, a predicted span on ignored text is not counted.

Only c1 and c2 are tuned, on the development split. The score used to choose them is the mean
of the per-language F1, so French and German count as much as English.

Usage:
    python -m src.uc2_extractor.training.train_crf --jobs 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Optional

from ..schema import LANGUAGES
from ..tagger import INFO_FILE, MIN_CONFIDENCE, MODEL_DIR, MODEL_FILE, CrfTagger, find_spans, token_features
from .common import TRAINING_DIR, read_jsonl
from .labels import IGNORE_TAG, spans_to_bio

SOURCES = (TRAINING_DIR / "converted" / "massive.jsonl", TRAINING_DIR / "converted" / "taskmaster2.jsonl")
DEV_SHARE = 15
GRID_C1 = (0.05, 0.2, 0.5)
GRID_C2 = (0.01, 0.1)
MAX_ITERATIONS = 120
MIN_LABEL_EXAMPLES = 20  # fewer training spans than this in a language: the label is off there


def is_dev(example: dict) -> bool:
    """True if this example belongs to the development split."""
    if example["source"] == "massive":
        return example["meta"]["partition"] != "train"
    dialogue = example["id"].rsplit("-", 1)[0]  # the last part is the turn number
    return hashlib.sha256(dialogue.encode("utf-8")).digest()[0] * 100 // 256 < DEV_SHARE


def load_split(sources: Iterable[Path] = SOURCES) -> tuple[list[dict], list[dict]]:
    examples = [e for path in sources for e in read_jsonl(path)]
    return [e for e in examples if not is_dev(e)], [e for e in examples if is_dev(e)]


def to_sequences(example: dict) -> list[tuple[list[dict], list[str]]]:
    """Features and tags of an example, cut into pieces at ignored tokens."""
    words, tags = spans_to_bio(example["text"], example["spans"], example.get("ignore"))
    pieces, xs, ys = [], [], []
    for x, y in zip(token_features(words, example["language"]), tags):
        if y == IGNORE_TAG:
            if xs:
                pieces.append((xs, ys))
            xs, ys = [], []
        else:
            xs.append(x)
            ys.append(y)
    if xs:
        pieces.append((xs, ys))
    return pieces


def labels_by_language(examples: Iterable[dict], minimum: int = MIN_LABEL_EXAMPLES) -> dict[str, list[str]]:
    """The labels each language has enough training spans for."""
    counts = Counter((e["language"], s["label"]) for e in examples for s in e["spans"])
    return {lang: sorted(label for (la, label), n in counts.items() if la == lang and n >= minimum)
            for lang in LANGUAGES}


def span_counts(gold: Iterable[dict], predicted: Iterable[dict], ignore: Iterable[dict] = ()) -> dict[str, list[int]]:
    """True positives, false positives and false negatives per label, on exact spans."""
    ignore = list(ignore)

    def on_ignored(s) -> bool:
        return any(s["start"] < p["end"] and p["start"] < s["end"] for p in ignore)

    g = {(s["start"], s["end"], s["label"]) for s in gold}
    p = {(s["start"], s["end"], s["label"]) for s in predicted if not on_ignored(s)}
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for item in g & p:
        counts[item[2]][0] += 1
    for item in p - g:
        counts[item[2]][1] += 1
    for item in g - p:
        counts[item[2]][2] += 1
    return counts


def rates(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn, "gold_spans": tp + fn}


def score(tagger, dev: list[dict]) -> dict:
    """Per language: precision, recall and F1 per label and for all labels pooled.

    Spans are filtered as in the app: only labels the language has training data for, at
    MIN_CONFIDENCE or above.
    """
    totals: dict[str, dict[str, list[int]]] = {lang: defaultdict(lambda: [0, 0, 0]) for lang in LANGUAGES}
    for e in dev:
        allowed = tagger.labels_for(e["language"])
        predicted = [{"start": s.start, "end": s.end, "label": s.label}
                     for s in find_spans(e["text"], e["language"], tagger)
                     if s.label in allowed and s.confidence >= MIN_CONFIDENCE]
        for label, row in span_counts(e["spans"], predicted, e.get("ignore", [])).items():
            for i in range(3):
                totals[e["language"]][label][i] += row[i]
    report = {}
    for lang, by_label in totals.items():
        report[lang] = {label: rates(*row) for label, row in sorted(by_label.items())}
        report[lang]["ALL"] = rates(*(sum(row[i] for row in by_label.values()) for i in range(3)))
    return report


def fit_one(params: tuple[float, float], model_path: Path, sources=SOURCES,
            max_iterations: int = MAX_ITERATIONS) -> dict:
    """Train with one c1 and c2, then score on the development split."""
    import sklearn_crfsuite

    c1, c2 = params
    train, dev = load_split(sources)
    pairs = [p for e in train for p in to_sequences(e)]
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=max_iterations,
                               model_filename=str(model_path))
    started = time.time()
    crf.fit([p[0] for p in pairs], [p[1] for p in pairs])
    tagger = CrfTagger(model_path, labels_by_language(train))
    report = score(tagger, dev)
    return {"c1": c1, "c2": c2, "seconds": round(time.time() - started), "report": report,
            "selection_score": round(sum(report[lang]["ALL"]["f1"] for lang in LANGUAGES) / len(LANGUAGES), 4),
            "model_path": str(model_path)}


def print_report(report: dict) -> None:
    print(f"{'lang':<5}{'label':<16}{'precision':>10}{'recall':>8}{'F1':>7}{'gold':>7}")
    for lang in LANGUAGES:
        for label, r in report[lang].items():
            print(f"{lang:<5}{label:<16}{r['precision']:>10.3f}{r['recall']:>8.3f}{r['f1']:>7.3f}{r['gold_spans']:>7}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    grid_dir = MODEL_DIR / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    train, dev = load_split()
    print(f"train {len(train)} examples, dev {len(dev)} examples", flush=True)
    grid = [(c1, c2) for c1 in GRID_C1 for c2 in GRID_C2]
    paths = [grid_dir / f"c1_{c1}_c2_{c2}.crfsuite" for c1, c2 in grid]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(fit_one, grid, paths, [SOURCES] * len(grid), [args.max_iterations] * len(grid)))
    for r in results:
        print(f"c1={r['c1']} c2={r['c2']} score={r['selection_score']} ({r['seconds']}s)", flush=True)

    best = max(results, key=lambda r: r["selection_score"])
    shutil.copyfile(best["model_path"], MODEL_FILE)
    info = {
        "model": "CRF, L-BFGS, sklearn-crfsuite", "c1": best["c1"], "c2": best["c2"],
        "max_iterations": args.max_iterations, "min_confidence": MIN_CONFIDENCE,
        "train_examples": len(train), "dev_examples": len(dev),
        "train_by_source": dict(Counter(e["source"] for e in train)),
        "dev_by_source": dict(Counter(e["source"] for e in dev)),
        "train_by_language": dict(Counter(e["language"] for e in train)),
        "dev_by_language": dict(Counter(e["language"] for e in dev)),
        "labels_by_language": labels_by_language(train),
        "grid": [{k: r[k] for k in ("c1", "c2", "selection_score", "seconds")} for r in results],
        "dev_report": best["report"],
        "data_sha256": {p.name: sha256(p) for p in SOURCES},
    }
    INFO_FILE.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nChosen: c1={best['c1']} c2={best['c2']}. Saved to {MODEL_FILE}")
    print("Development split, exact spans, as filtered in the app:")
    print_report(best["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
