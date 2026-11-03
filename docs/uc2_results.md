# UC2 Travel Enquiry Extractor: Results

First version: 30 October 2026. Last updated: 3 November 2026.

## 1. What was built

The extractor turns a free-text travel enquiry in English, French or German into a validated JSON client profile. It is rule-based. An optional model layer (section 4) can fill some label fields that the rules leave empty.

- **Schema** (`src/uc2_extractor/schema.py`): one pydantic definition of the profile.
  - Fields: destination, dates or duration, travellers (party size, adults, children, ages), budget (amount, currency, per trip or per person, band), accommodation, flights, interests, dietary needs, occasion, special requests and must-haves, and language.
  - Every field can be empty. Every value comes with the piece of text that supports it.
  - Missing, ambiguous or assumed values are listed as flags.
- **Date parser** (`dates.py`): handles month names in all three languages, numeric dates, ranges and trip lengths. A date such as 3/4 is flagged with both readings and is never guessed. A year is never added if it is not written.
- **Rule-based extractors** (`extract.py`, `lexicon.py`): keyword lists in three languages, number words, a simple check for negation ("no shopping", "kein", "pas de"), and budget parsing for 20,000 / 20.000 / 20 000 / 4,00,000.
- **Language detection** (`text.py`): counts common function words. It needs no extra package.
- **Evaluation** (`evaluate.py`, `metrics.py`): exact and lenient precision, recall and F1 per field and per language, and a table of every error.
- **Optional model layer** (`semantic.py`). See section 4.
- **Demos:**
  - Command line (`app/uc2_extract_cli.py`).
  - Browser (`app/uc2_web_demo.py`), with these features:
    - one enquiry or several (one per line);
    - rules only or rules + model;
    - a "rule" or "model" tag with confidence on every value;
    - the scores, with the real number kept apart from the optimistic ones.
  - If the model files are missing, the browser demo runs rules only and says so.
- **Review and correct screen** (in the browser demo):
  - After extracting one enquiry, a reviewer can edit, remove or add values and click "Save corrections".
  - Each correction is saved as one line in `data/uc2_inquiries/corrections/corrections.jsonl` (not committed), with the enquiry text, the extracted profile, the corrected profile, the language and a timestamp.
  - Corrected values must pass the same schema checks as extracted ones. The extracted profile is recomputed on the server, not taken from the browser.
  - `python -m src.uc2_extractor.corrections --to-csv <folder>` turns the saved corrections into the held-out CSV format, marked "from corrections, needs review".
  - The page warns that real client enquiries must not be pasted in until the team agrees how to store them.
- **Local API** (`app/uc2_api.py`, FastAPI):
  - `POST /extract` returns the validated profile with flags, source text and rule/model tags.
  - `GET /health` says whether the model is loaded.
  - Texts are limited to 20,000 characters and bad input gets short errors. Enquiry text is never logged or echoed back.
  - FastAPI's documentation page is at `/docs`.
- **Tests:** 433 pytest tests, all passing. They cover the schema, date handling, tokenizer offsets, metrics, extractor behaviour, the model layer (including negation), the demo (including that the page never inserts HTML), the API (including bad input and the missing-model fallback) and saving and converting corrections.

## 2. Data and licences

- **Gold set:** 60 synthetic enquiries (20 per language) written for this project. No real customer messages were used. The files are in `data/uc2_inquiries/gold/` and are not committed to git. See `docs/uc2_labelling_guide.md`.
- **Taskmaster-2:** released under CC BY 4.0 according to its official README. It was **not used** in this first version, because its hotel and flight dialogues do not have labels for most of the fields in our schema. If it is used later, its attribution must be kept.
- **Budget bands:** under 10k, 10k-25k, 25k-50k and over 50k USD per trip. These are **PLACEHOLDER values**, not agreed business values. They are set in `src/uc2_extractor/budget_bands.toml`. Currencies are never converted, so EUR, GBP, INR and CHF budgets get no band.

## 3. How to reproduce

```
.venv\Scripts\activate
python -m pytest
python -m src.uc2_extractor.evaluate --show-all
python -m src.uc2_extractor.evaluate --gold-dir data/uc2_inquiries/heldout --model
python app/uc2_extract_cli.py --text "My wife and I want 2 weeks in Kerala in March."
python app/uc2_web_demo.py        (then open http://localhost:8000)
python app/uc2_api.py             (then open http://127.0.0.1:8001/docs)
python -m src.uc2_extractor.corrections --to-csv data/uc2_inquiries/corrections/as_labels
```

The model files are downloaded once with:

```
python -c "from src.uc2_extractor.semantic import download_model; download_model()"
```

## 4. Results

Scores are exact-match precision (P), recall (R) and F1. All fields are pooled for each language. "Values found where gold was empty" counts fields where the extractor produced a value although the gold label was empty.

### Held-out, rules only: the real number

This is the most honest result so far. On 3 November 2026 a new held-out set of 45 synthetic enquiries (15 per language) was written in a different style from the gold set: long rambling emails, very short notes, typos, missing details and unusual wording. All labels were written and the files were frozen before the extractor was run on them. The extractor (rules only, as of commit `3931a86`) was then scored on it once.

The files are in `data/uc2_inquiries/heldout/` (not committed). `FROZEN.txt` there records their SHA-256 fingerprints:
- enquiries.csv `7FE2E1A7787AF48A323D2009DE6096E4CAF7A5A36EAE5264A3FE670B448217AE`
- labels.csv `631A231689F6CF8A244E46321FB99EB0C242A796B2741CAA36C74FDB88E2FDD8`

| Language | Enquiries | P | R | F1 | Lenient F1 | Values found where gold was empty |
|---|---|---|---|---|---|---|
| English | 15 | 0.92 | 0.71 | 0.81 | 0.81 | 2 |
| French | 15 | 0.97 | 0.81 | 0.88 | 0.88 | 3 |
| German | 15 | 0.96 | 0.78 | 0.86 | 0.86 | 4 |

There were 117 field errors: 100 missed, 5 wrong and 12 extra. Precision stays high, so the extractor rarely invents values. Recall is the weak point: it misses values written in ways the rules do not cover.

| Field | P | R | F1 | | Field | P | R | F1 |
|---|---|---|---|---|---|---|---|---|
| language | 1.00 | 0.93 | 0.97 | | budget_min | 1.00 | 0.96 | 0.98 |
| destination | 0.93 | 0.91 | 0.92 | | budget_max | 1.00 | 0.96 | 0.98 |
| start_date | 0.83 | 0.77 | 0.80 | | currency | 1.00 | 0.96 | 0.98 |
| end_date | 1.00 | 0.56 | 0.71 | | budget_basis | 0.92 | 0.79 | 0.85 |
| duration | 0.97 | 0.90 | 0.93 | | budget_band | 1.00 | 1.00 | 1.00 |
| dates_ambiguous | 1.00 | 1.00 | 1.00 | | accommodation | 1.00 | 0.57 | 0.73 |
| party_size | 0.88 | 0.66 | 0.75 | | flights | 0.92 | 0.69 | 0.79 |
| adults | 0.94 | 0.52 | 0.67 | | interests | 0.92 | 0.61 | 0.73 |
| children | 1.00 | 0.75 | 0.86 | | dietary | 1.00 | 0.60 | 0.75 |
| ages | 0.89 | 0.44 | 0.59 | | occasion | 1.00 | 0.40 | 0.57 |
| | | | | | special_requests | 1.00 | 0.41 | 0.58 |

Errors by type (117 in total):
- 40 in accommodation, interests, special requests, occasion and dietary. The wording is outside the word lists, for example "used to be a maharaja's house", "a desert camp night under canvas", "tied the knot", "noces de perle", "my husband's 50th", "doesn't eat meat", "a car with a driver", "someone to meet us at the airport" and "mad about birds".
- 35 in travellers. Groups are described indirectly, for example "Dad, me and my partner", "me + 3 mates", "deux couples d'amis", "2 Erw.", "wir (beide 75)", "j'ai 63 ans" and "twins who'll be 12".
- 19 in dates:
  - A month written without a preposition ("März,", "Novembre,").
  - Typos ("Novmber", "debut mars").
  - An unambiguous end date taken as the start date when the start date was ambiguous (5/11 ... 19/11).
  - A year not carried over within a range.
  - "February or March" read as a start month.
  - "Dez/Jan".
- 7 in destination. Places that are mentioned but not visited were picked up (Bhutan, Delhi and Agra in a "we did that before" sentence; Delhi as an arrival airport), and typos were missed (Sri Lnka, Maledieven).
- 7 in budget fields. A budget written in words ("fifteen thousand pounds") was missed entirely, and some per-trip or per-person wording was missed.
- 6 in flights and 3 in language (very short notes such as "Kerala 6/1-20/1, 2 Erw. + 2 Kinder, Hausboot!").

Any later score on this set, after rules or models are changed with these errors in view, must be labelled "tuned on this set, optimistic".

### Held-out after general rule improvements: tuned on this set, optimistic

After the run above, the rules were improved as general rules, not as copies of single enquiries:
- travellers given as friends or couples ("me + 3 mates", "deux couples", "zwei Paare", "three of us", "2 Erw.");
- more ways of giving ages ("j'ai 63 ans", "my dad is 78", "beide 75"), without counting "40 ans d'enseignement";
- misspelt month names, months standing alone, "Dez/Jan", "February or March" as a choice, a year written once in a range, and an end date when the start is ambiguous;
- a narrower negation scope, and negative wording that names a diet ("doesn't eat meat", "isst kein Fleisch");
- budgets written in words;
- misspelt destination names.

Each rule has tests in English, French and German (362 tests in total). The labels were not changed.

| Language | P | R | F1 | Change from the real number |
|---|---|---|---|---|
| English | 0.94 | 0.80 | 0.87 | +0.06 |
| French | 0.98 | 0.86 | 0.91 | +0.03 |
| German | 0.99 | 0.89 | 0.93 | +0.07 |

There are now 74 field errors (63 missed, 2 wrong, 9 extra), down from 117:

| Error group | Before | After |
|---|---|---|
| Accommodation, interests, special requests, occasion, dietary | 40 | 34 |
| Travellers | 35 | 19 |
| Dates | 19 | 5 |
| Flights | 6 | 6 |
| Destination | 7 | 3 |
| Budget | 7 | 4 |
| Language | 3 | 3 |

These scores are **optimistic**, because the rules were chosen with this set's errors in view. The fair test of these changes is the second held-out set, which colleagues are writing and which will be scored once at the end. On the original gold set the scores did not change (English 0.96, French 0.98, German 0.98; 42 errors), so the new rules did not break anything there.

### Optional model layer: rules only against rules plus model (tuned on this set, optimistic)

**The model.**
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, the quantised ONNX file `onnx/model_quint8_avx2.onnx` (113 MiB). It runs with onnxruntime on the CPU, with no PyTorch.
- The licence is Apache-2.0 according to the model card on Hugging Face. The licences of its training data were not checked.
- The files are kept in `models/uc2_minilm/`, which is not committed.

**How it is used.**
- The rules run first. The model only fills interests, occasion, accommodation, dietary or special requests when the rules left that field empty.
- It compares short pieces of the enquiry with a few example phrases per label, written by hand in English, French and German.
- Any clause that contains a negation word is skipped, including the shorter windows cut from it.
- Each value is marked "rule" or "model". Model values carry the similarity score as their confidence. This is a matching score, not a probability.
- Rules only is still the default (`model=False`).

**Choosing the threshold (on the gold set only).**

| Mode on the gold set | Precision (5 fields) | Recall (5 fields) | F1 (5 fields) | False positives |
|---|---|---|---|---|
| Rules only | 0.99 | 0.93 | 0.96 | 1 |
| Rules + model at 0.60 | 0.72 | 0.94 | 0.81 | 73 |
| Rules + model at 0.75 | 0.94 | 0.94 | 0.94 | 11 |
| Rules + model at 0.85 | 0.99 | 0.93 | 0.96 | 2 |
| Rules + model at 0.88 | 0.99 | 0.93 | 0.96 | 1 |

On the gold set the model never beat rules only. The rules were built on this set and already cover its wording. The threshold was set to 0.88, the lowest value at which the model added no false positive on the gold set. This makes the model cautious.

**Result on the first held-out set at threshold 0.88.** Both rows are tuned on this set, optimistic.

| Language | Rules only F1 | Rules + model F1 | Rules only P | Rules + model P |
|---|---|---|---|---|
| English | 0.87 | 0.87 | 0.94 | 0.95 |
| French | 0.91 | 0.92 | 0.98 | 0.98 |
| German | 0.93 | 0.94 | 0.99 | 0.99 |

| Field | Rules only (correct / missed / extra) | Rules + model (correct / missed / extra) |
|---|---|---|
| accommodation | 12 / 9 / 0 | 14 / 7 / 0 |
| occasion | 4 / 6 / 0 | 5 / 5 / 0 |
| special_requests | 7 / 10 / 0 | 9 / 8 / 0 |
| interests | 23 / 13 / 2 | 23 / 13 / 2 |
| dietary | 14 / 1 / 0 | 14 / 1 / 0 |

The model fixed 5 values and added no wrong values:
- "frisch verheiratetes Paar" → honeymoon
- "a car with a driver" → private driver
- "quelqu'un qui conduit" → private driver
- "lodge au cœur de la jungle" → safari lodge
- "hôtels de charme" → boutique hotel

Errors fell from 74 to 71, and precision did not drop.

**Plain conclusion.** At a threshold chosen fairly on the gold set, the model helps a little and safely, but not much. It fixed 5 of the 34 wording errors that remain after the rule improvements, and none of the interests errors. A lower threshold would catch more, but on the gold set it added many wrong values (73 at 0.60), so it is not used. The real test is the second held-out set.

**Two things that make even these 5 fixes look better than they are:**
- **The example phrases were written after seeing this set's errors.** Several are close to its wording:
  - "a car with our own driver" (held-out HE05: "a car with a driver");
  - "ein kleines, persönliches Hotel mit Charme" (HD10);
  - "a lodge in the jungle" (HF13);
  - "un petit hôtel de charme" (HF15).

  The 5 fixes are therefore not independent evidence that the model generalises.
- **On new wording the model is mostly below the threshold.** A quick check on phrasings that appear in no test set and no example phrase found the right label ranked first, but under 0.88:
  - "keen birders" → wildlife 0.70;
  - "pampered with massages" → wellness 0.87;
  - "make proper curries" → food and cooking 0.76.

  At 0.88 the model therefore adds almost nothing on fresh text. Choosing a better threshold needs a small separate tuning set: not the gold set (the rules already cover it) and not a held-out set (that would be tuning on the test).

### Earlier results on the gold set (60 enquiries)

The gold set was written by the same person who wrote the rules, and the rules were later fixed on it. Even its run 1 is therefore less fair than the held-out real number above. The results are kept here for the record.

#### Gold set run 1: before any fixes

| Language | Enquiries | P | R | F1 | Lenient F1 | Values found where gold was empty |
|---|---|---|---|---|---|---|
| English | 20 | 0.92 | 0.84 | 0.88 | 0.89 | 11 |
| French | 20 | 0.95 | 0.89 | 0.92 | 0.93 | 3 |
| German | 20 | 0.96 | 0.84 | 0.90 | 0.90 | 5 |

There were 135 field errors in total: 93 missed, 18 wrong and 24 extra.

#### Gold set run 2: after fixing the errors seen in run 1 (optimistic)

| Language | Enquiries | P | R | F1 | Lenient F1 | Values found where gold was empty |
|---|---|---|---|---|---|---|
| English | 20 | 0.97 | 0.95 | 0.96 | 0.96 | 6 |
| French | 20 | 0.99 | 0.97 | 0.98 | 0.98 | 2 |
| German | 20 | 1.00 | 0.96 | 0.98 | 0.98 | 1 |

There were 42 field errors in total: 32 missed, 1 wrong and 9 extra.

**Run 2 is not a fair test.** The rules were changed after looking at the run 1 errors on the same 60 enquiries. Between the runs, one gold label was also corrected (E15: adults = 12, to match how "three friends" was labelled). Run 1 is the better estimate of how the extractor does on new text. Neither run says anything about real customer enquiries.

#### Gold set F1 by field (all languages)

| Field | Run 1 F1 | Run 2 F1 |
|---|---|---|
| language | 1.00 | 1.00 |
| destination | 0.95 | 0.98 |
| start_date | 0.96 | 0.98 |
| end_date | 0.93 | 1.00 |
| duration | 0.84 | 0.91 |
| dates_ambiguous | 0.55 | 0.91 |
| party_size | 0.75 | 0.99 |
| adults | 0.76 | 0.89 |
| children | 0.67 | 0.87 |
| ages | 0.54 | 1.00 |
| budget_min | 0.90 | 1.00 |
| budget_max | 0.98 | 1.00 |
| currency | 1.00 | 1.00 |
| budget_basis | 0.86 | 1.00 |
| budget_band | 1.00 | 1.00 |
| accommodation | 0.93 | 0.97 |
| flights | 0.86 | 0.98 |
| interests | 0.93 | 0.96 |
| dietary | 0.94 | 0.98 |
| occasion | 0.84 | 1.00 |
| special_requests | 0.90 | 0.92 |

The full tables, including results per field and per language, are written to `data/uc2_inquiries/eval_output/`.

In both runs, all evidence spans matched the enquiry text exactly.

## 5. What works

- **Precision is high everywhere.** On new text the extractor rarely invents a value: precision on the first held-out run was 0.92 to 0.97.
- **Budgets:** amounts, currency and the placeholder band were reliable on both sets. This includes European and Indian number formats, "up to" / "bis zu" / "jusqu'à", and budgets written in words.
- **Ambiguous numeric dates** (3/4, 02.01.) are flagged with both readings instead of guessed.
- **Clear statements** of destination, dates, trip length, party size and dietary needs work in all three languages.
- **Every value comes with the text it was taken from.** In every run, all evidence spans matched the enquiry text exactly.
- **The model layer is safe at its threshold:** it added no wrong values on the held-out set. Rules only remains the default.

## 6. What fails

The remaining **71 errors** on the first held-out set, with rules + model (tuned on this set, optimistic), are:

| Group | Errors | Typical examples |
|---|---|---|
| Accommodation, interests, special requests, occasion, dietary | 31 (29 missed, 2 extra) | Wording outside the word lists that the model scores below the threshold, e.g. "mad about birds", "old buildings", "under canvas", "tied the knot", "noces de perle", "someone to meet us at the airport" |
| Travellers | 19 (15 missed, 2 wrong, 2 extra) | Adults implied by a group or family ("group of 12 colleagues", "me, my wife, our twins"); ages written as "twins who'll be 12"; "(30 ans !)" read as an age; "2 or 3 of us" read as 3 |
| Flights | 6 | Heathrow not linked to London; "partirons de Lyon", "Départ de Genève"; "sort our own flights"; "pro Person ohne Flüge" wrongly read as "flights not needed" |
| Dates | 5 | Trip length missed when another length is also mentioned ("a day on a houseboat"); "une dizaine de nuits", "une douzaine de jours"; a part-length taken as the trip ("eine Woche in Udaipur"); "vacances de février" |
| Budget | 4 | Per trip / per person wording such as "all in", "for the two of us", "par persone" (typo) |
| Destination | 3 | Places mentioned but not visited ("we did Delhi and Agra before", Delhi as an arrival airport); Annapurna not in the place list |
| Language | 3 | Very short notes with no clear words or letters |

Other known problems:
- **"Two of us are vegetarian"** is read as a couple (gold P04-en).
- **The demo's French example works now** ("Nous sommes deux adultes et deux enfants" gives 4 travellers), but short notes with only place names still get no language.

## 7. Honest limits

- **All the test text is synthetic.** It was written for this project, not taken from real customer messages, so none of these results show performance on real Kenexoft enquiries.
- **The only fair number is the first held-out run** (0.81 / 0.88 / 0.86). Every later number was produced after looking at that set's errors and is optimistic.
- **The same person wrote the rules, the example phrases for the model, and both test sets.** That makes all scores likely higher than on text written by others.
- **The model's example phrases were written after seeing the held-out errors.** Some are close to held-out wording, so its 5 fixes there are not independent evidence (section 4).
- **The model threshold was chosen on the gold set**, where the rules already cover the wording. As a result the model is very cautious, and on fresh wording it mostly stays below the threshold.
- **French and German texts in both sets** were not written by native speakers and have not been checked by one.
- **No second annotator has labelled anything yet**, so there is no agreement figure.
- **Results in one language do not show the same quality in another.**
- **The evaluation compares values, not character spans.**
- **Labels come from a fixed list**, so anything outside the list is never extracted.
- **Rule values have no confidence.** Model confidences are similarity scores, not probabilities.
- **The budget bands are placeholders.**
- **No real data has gone through the API or the review screen yet.** Corrections are stored in a plain local file with no access control, deletion or retention rules. The team must agree how client enquiries may be stored before real ones are used.
- **Corrections are not labelled data yet.** They are one reviewer's edits. They need the same checking as the gold set before they can be used for evaluation, and they must never be mixed into a held-out set.
- **The API has been tested only on this computer** (127.0.0.1), not deployed. Its `/docs` page loads scripts from a public CDN, and it was not checked in a real browser during development.

## 8. What is left to do

**By hand (project owner):**
1. Collect held-out set 2 from colleagues in `data/uc2_inquiries/heldout2/`, using the template and instruction sheet there. It is then labelled, frozen with SHA-256 fingerprints, and scored once at the end, for rules only and rules + model.
2. Check the labels:
   - **Gold set:** rewrite the 6 "rewrite" enquiries, and check the enquiries marked "check labels".
   - **First held-out set:** check all 45, starting with the judgement calls listed in their review notes.
   - Record progress in the `checked_by` and `checked_on` columns.
3. Have a fluent speaker check the French and German text in both sets (40 gold, 30 held-out).
4. Arrange the second annotator: 12 gold enquiries and 9 held-out enquiries are marked.
5. Agree the real budget bands.
6. Agree with the team how real client enquiries may be stored before anyone pastes one into the demo or the API.

**Possible technical next steps (only after the above):**
- Choose the model threshold on a small separate tuning set (not gold, not held-out), and say so in this note.
- Add rules for adults implied by groups or families, "already visited" places, and departure cities such as Heathrow.
- Consider the CRF step only if French or German stay clearly weaker on held-out set 2.
