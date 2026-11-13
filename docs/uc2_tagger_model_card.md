# UC2 Slot Tagger: Model Card

## What it is

A conditional random field (CRF) that puts a BIO tag on each token of an enquiry. It finds spans such as a destination or a date. It was trained for this project on 6 November 2026.

Files, not stored in this repository: `models/uc2_crf/crf_tagger.crfsuite` (the model, about 0.6 MB) and `models/uc2_crf/crf_tagger.json` (its settings and scores).

To train it again:

```
python -m src.uc2_extractor.training.train_crf --jobs 4
```

## Intended use

- A helper that fills fields the hand-written rules left empty, in the `rules_tagger` mode.
- Every value it adds is marked `crf`, with a probability, and shows the text it came from.
- A person checks the result on the review screen.

## Not intended for

- Replacing the rules. On its own it is much weaker than they are.
- French or German fields other than dates. It has no training data for them.
- Decisions with no person checking them.
- Real client messages, until the team agrees how to handle them.

## Training data

13,965 examples from MASSIVE (English, French, German) and Taskmaster-2 hotels and flights (English). Full details are in `docs/uc2_datasheet.md`. No synthetic data and no client messages were used.

## Licences

- Training data: CC BY 4.0, with attribution in `NOTICE`.
- sklearn-crfsuite 0.5.0 and python-crfsuite 0.9.12: MIT, according to their package details.
- Not checked: how CC BY 4.0 applies to trained weights. The attribution is kept either way.

## How it works

- Features of each token: lowercase word, 2 and 3 letter prefix and suffix, word shape, digit flag, the two words on each side, and a language tag.
- Training: L-BFGS, 120 iterations at most.
- Only c1 and c2 were tuned, on the development split. The score was the mean of the per-language F1.

| c1 | c2 | Score |
|---|---|---|
| 0.05 | 0.01 | 0.761 |
| 0.05 | 0.1 | 0.771 |
| 0.2 | 0.01 | 0.770 |
| **0.2** | **0.1** | **0.773** |
| 0.5 | 0.01 | 0.772 |
| 0.5 | 0.1 | 0.771 |

Three things limit what reaches a client profile. None of them was tuned:

- A label can fill a field only in a language with at least 20 training spans for it. That means nine labels in English and DATE only in French and German. If the language cannot be detected, DATE only.
- A span needs a confidence of at least 0.5, fixed in advance. Confidence is the lowest marginal probability among the span's tokens. It is not calibrated.
- A span in a negated clause is dropped.

Spans become values through the existing code for dates, numbers and currency. A date the code cannot read is flagged for a person and left empty.

## Results on the development split

Exact spans, with the filters above except negation. This split comes from the same public sources as the training data, so these scores say nothing about client emails.

| Language | Label | Precision | Recall | F1 | Gold spans |
|---|---|---|---|---|---|
| en | ACCOMMODATION | 0.585 | 0.554 | 0.569 | 112 |
| en | ADULTS | 0.917 | 1.000 | 0.957 | 22 |
| en | CHILDREN | 1.000 | 1.000 | 1.000 | 7 |
| en | DATE | 0.794 | 0.732 | 0.762 | 1,246 |
| en | DESTINATION | 0.933 | 0.899 | 0.916 | 664 |
| en | DURATION | 0.833 | 0.870 | 0.851 | 23 |
| en | FLIGHT | 0.916 | 0.914 | 0.915 | 419 |
| en | ORIGIN | 0.954 | 0.916 | 0.935 | 273 |
| en | PARTY | 0.975 | 0.951 | 0.963 | 82 |
| en | All labels | 0.860 | 0.818 | 0.839 | 2,848 |
| fr | DATE | 0.710 | 0.648 | 0.678 | 219 |
| de | DATE | 0.837 | 0.769 | 0.802 | 221 |

ADULTS and CHILDREN have very few gold spans, so their scores mean little.

## Results on held-out set 1

The tagger never saw this set, but the rules were tuned on it, so the comparison favours the rules. All 45 enquiries are synthetic.

| Mode | en | fr | de | Field errors |
|---|---|---|---|---|
| Rules only | 0.87 | 0.91 | 0.93 | 74 |
| Rules + tagger | 0.87 | 0.91 | 0.93 | 75 |

The tagger added 1 correct value and 2 wrong values. On its own, on the fields it covers, it scored F1 0.40 in English, 0.48 in French and 0.42 in German, against 0.88, 0.97 and 1.00 for the rules. The full tables are in `docs/uc2_results.md`.

So on this set the tagger did not improve the system. It can find places that are not in the word lists, and it gives later models a trained baseline to beat.

## Limits

- Trained on voice commands and transcribed speech, not client emails.
- English only, apart from dates.
- It cannot find ages, budgets, interests, dietary needs, occasions or special requests.
- Its probabilities are not calibrated.
- It has not been tested on real enquiries or on held-out set 2.

## Second tagger (v2)

- **Files:** `models/uc2_crf/crf_tagger_v2.crfsuite`, `crf_tagger_v2.json` and `crf_tagger_v2_labels.joblib` (not stored in this repository). The first tagger is kept unchanged beside it.
- **Data:** the same public data plus the training part of 900 synthetic enquiries (14,685 training examples in all). See section 9 of `docs/uc2_datasheet.md`.
- **Labels:** all 15 labels in English, French and German.
- **Settings:** c1 = 0.05, c2 = 0.1, chosen on the development split with the same 6-value grid.
- **Development F1, all labels:** en 0.847, fr 0.828, de 0.883. The development split now includes synthetic enquiries, which are easier than real ones.
- **Span-to-label classifier:** one small text classifier per label (character n-grams, logistic regression) turns a span such as "seeing big cats in the wild" into its field label. It is only used when the word lists find nothing, and only when it is at least 0.5 sure. On the development split it got these right: accommodation 100% of 44, dietary 100% of 35, flight 100% of 456, interest 99% of 94, occasion 96% of 22, special_request 100% of 38. These are mostly synthetic spans, so the numbers are optimistic.
- **Results:** see "Second tagger (public + synthetic data) compared with the first" in `docs/uc2_results.md`. On its own it is far better than the first tagger. Added to the rules, it did not reduce errors on the gold set or held-out set 1, and some fields got slightly worse.
- **Mode:** `rules_tagger_v2` in the API, "rules + new tagger" in the demo.
