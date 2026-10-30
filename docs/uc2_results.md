# UC2 Travel Enquiry Extractor: First Version Results

Date: 30 October 2026

## 1. What was built

The first version of the extractor turns a free-text travel enquiry in English, French or German into a validated JSON client profile. It uses rules only, with no machine-learning model.

- **Schema** (`src/uc2_extractor/schema.py`): one pydantic definition of the profile.
  - Fields: destination, dates or duration, travellers (party size, adults, children, ages), budget (amount, currency, per trip or per person, band), accommodation, flights, interests, dietary needs, occasion, special requests and must-haves, and language.
  - Every field can be empty. Every value comes with the piece of text that supports it.
  - Missing, ambiguous or assumed values are listed as flags.
- **Date parser** (`dates.py`): handles month names in all three languages, numeric dates, ranges and trip lengths. A date such as 3/4 is flagged with both readings and is never guessed. A year is never added if it is not written.
- **Rule-based extractors** (`extract.py`, `lexicon.py`): keyword lists in three languages, number words, a simple check for negation ("no shopping", "kein", "pas de"), and budget parsing for 20,000 / 20.000 / 20 000 / 4,00,000.
- **Language detection** (`text.py`): counts common function words. It needs no extra package.
- **Evaluation** (`evaluate.py`, `metrics.py`): exact and lenient precision, recall and F1 per field and per language, and a table of every error.
- **Command-line demo** (`app/uc2_extract_cli.py`).
- **Tests:** 241 pytest tests, all passing. They cover the schema, date handling, tokenizer offsets, metrics and extractor behaviour.

## 2. Data and licences

- **Gold set:** 60 synthetic enquiries (20 per language) written for this project. No real customer messages were used. The files are in `data/uc2_inquiries/gold/` and are not committed to git. See `docs/uc2_labelling_guide.md`.
- **Taskmaster-2:** released under CC BY 4.0 according to its official README. It was **not used** in this first version, because its hotel and flight dialogues do not have labels for most of the fields in our schema. If it is used later, its attribution must be kept.
- **Budget bands:** under 10k, 10k-25k, 25k-50k and over 50k USD per trip. These are **PLACEHOLDER values**, not agreed business values. They are set in `src/uc2_extractor/budget_bands.toml`. Currencies are never converted, so EUR, GBP, INR and CHF budgets get no band.

## 3. How to reproduce

```
.venv\Scripts\activate
python -m pytest
python -m src.uc2_extractor.evaluate --show-all
python app/uc2_extract_cli.py --text "My wife and I want 2 weeks in Kerala in March."
```

## 4. Results

Scores are exact-match precision (P), recall (R) and F1. All fields are pooled for each language. "Values found where gold was empty" counts fields where the extractor produced a value although the gold label was empty.

### Run 1: before any fixes (the fairer number)

| Language | Enquiries | P | R | F1 | Lenient F1 | Values found where gold was empty |
|---|---|---|---|---|---|---|
| English | 20 | 0.92 | 0.84 | 0.88 | 0.89 | 11 |
| French | 20 | 0.95 | 0.89 | 0.92 | 0.93 | 3 |
| German | 20 | 0.96 | 0.84 | 0.90 | 0.90 | 5 |

There were 135 field errors in total: 93 missed, 18 wrong and 24 extra.

### Run 2: after fixing the errors seen in run 1 (optimistic)

| Language | Enquiries | P | R | F1 | Lenient F1 | Values found where gold was empty |
|---|---|---|---|---|---|---|
| English | 20 | 0.97 | 0.95 | 0.96 | 0.96 | 6 |
| French | 20 | 0.99 | 0.97 | 0.98 | 0.98 | 2 |
| German | 20 | 1.00 | 0.96 | 0.98 | 0.98 | 1 |

There were 42 field errors in total: 32 missed, 1 wrong and 9 extra.

**Run 2 is not a fair test.** The rules were changed after looking at the run 1 errors on the same 60 enquiries. Between the runs, one gold label was also corrected (E15: adults = 12, to match how "three friends" was labelled). Run 1 is the better estimate of how the extractor does on new text. Neither run says anything about real customer enquiries.

### F1 by field (all languages)

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

- Language detection was correct for all 60 enquiries in both runs.
- Currency, budget amounts and the placeholder band were reliable on this set, including European number formats and "up to" / "bis zu" / "jusqu'à".
- Ambiguous numeric dates (3/4, 02.01.) are flagged with both readings instead of guessed.
- Clear statements of party size, dates, dietary needs, occasions and accommodation types work in all three languages.
- Values are not invented for empty fields in most cases. Run 1 found values where the gold was empty 19 times across 1,260 field values, and run 2 found 9.

## 6. What fails (errors still present in run 2)

- **Children and adults implied rather than stated.** Examples are "kids aged 9 and 13" with no number before "kids", "three friends" when a group count is also given, and "Nous serons deux", "on est 2 potes" or "my parents" without the word adults.
- **Part-lengths of a trip.** "One week in Sri Lanka followed by a few days in the Maldives" gives 1 week, and "add 3 nights in Jaipur" gives 3 nights. These are only parts of the trip.
- **Dates that are not travel dates.** A phone-call date and an email "Sent" date were flagged as ambiguous travel dates (E14). "Turns 60 in March" gave March as the start date (E17).
- **Paraphrases outside the word lists.** Examples: "vehicle with driver", "something special arranged", "Hôtels de charme historiques", "small boutique properties", "gutes Essen", "une dizaine de jours", "Yoga-Retreat-Woche".
- **Places outside the place list** (Pangong Lake), and Mumbai picked up as a destination in "une famille de Mumbai".
- **One problem made worse by a fix.** "Two of us are vegetarian" is now read as a couple (adults = 2) in a group of 8 (P04-en).

## 7. Honest limits

- The gold set is small (60 enquiries) and synthetic. It is a first evaluation set, not proof that the extractor is ready for production.
- The same person wrote the rules and the gold set, so the scores are likely higher than they would be on text written by someone else. Six enquiries are marked "rewrite" for this reason.
- The French and German enquiries were not written by native speakers and have not yet been checked by one.
- The second annotator has not labelled anything yet, so there is no agreement figure.
- Good results in one language do not show the same quality in another, and none of these results show performance on real Kenexoft enquiries.
- The evaluation compares values, not character spans, because the gold set records the expected values.
- Labels come from a fixed list, so anything outside the list is not extracted.
- The extractor gives no confidence scores.
- The budget bands are placeholders.

## 8. Suggested next steps

1. Rewrite the six "rewrite" enquiries, have a fluent speaker check the French and German, and complete the second annotation (12 enquiries).
2. Write a new set of enquiries that is kept apart and never used to change the rules. Report that score as the main result.
3. Agree the real budget bands and replace the placeholders.
4. Only if the rules are clearly not good enough in French or German on fresh data, consider the CRF step that was planned for later.
