# Kenexoft AI/ML Internship Project

This repository holds two separate AI/ML use cases built during an internship at Kenexoft, a luxury South Asia travel company.

## Use Case 1: Hotel Rate Prediction and Demand Analysis

Builds a reference hotel nightly rate from historical booking data and tests whether booking-demand signals make that reference rate more accurate. It uses the public Hotel Booking Demand dataset, which covers two hotels in Portugal from 2015 to 2017.

The recorded rate (adr) is not a live quote. Results on this dataset say nothing about how the model would perform on South Asian luxury hotels.

## Use Case 2: Multilingual Travel Inquiry Information Extractor

Pulls key details out of travel inquiries written in English, French or German: destination, dates or duration, travellers, budget, accommodation preferences, flight requirements and special requests. It shows the text that supports each value and flags information that is missing or unclear.

The first version is rule-based. It returns a validated JSON profile with the fields destination, dates or duration, travellers, budget (band plus original amount and currency), accommodation, flights, interests, dietary needs, occasion, special requests and language. Fields that are not found stay empty, and missing or ambiguous details are flagged.

It is evaluated on a small synthetic gold set written for this project (60 enquiries, 20 per language). Results and limits are in `docs/uc2_results.md`. The labelling rules are in `docs/uc2_labelling_guide.md`. The budget bands in `src/uc2_extractor/budget_bands.toml` are placeholders.

A trained slot tagger (a CRF) can now fill fields the rules leave empty. It is trained on the public MASSIVE and Taskmaster-2 datasets, knows nine labels in English, and only dates in French and German. On the held-out set it did not improve the results; see `docs/uc2_results.md`. Details are in `docs/uc2_tagger_model_card.md`, `docs/uc2_datasheet.md` and `docs/uc2_label_scheme.md`. The attribution for both datasets is in `NOTICE`.

Try it:

```
python app/uc2_extract_cli.py --text "My wife and I want 2 weeks in Kerala in March, budget USD 20,000."
python -m src.uc2_extractor.evaluate
python -m pytest
```

The tagger is not in this repository. To train it, put the datasets in place, then run the converters and the trainer:

```
python -m src.uc2_extractor.training.convert_massive
python -m src.uc2_extractor.training.convert_taskmaster
python -m src.uc2_extractor.training.train_crf --jobs 4
python -m src.uc2_extractor.training.compare_tagger
```

### Browser demo and corrections

```
python app/uc2_web_demo.py        (then open http://localhost:8000)
```

After extracting an enquiry you can correct the values and save them. Corrections go to `data/uc2_inquiries/corrections/corrections.jsonl`, which is not committed. Do not paste real client enquiries until the team agrees how to store them. To turn saved corrections into held-out style CSV files for review, run:

```
python -m src.uc2_extractor.corrections --to-csv data/uc2_inquiries/corrections/as_labels
```

### Local API

A small FastAPI app wraps the extractor. It listens on this computer only and never logs or echoes enquiry text.

```
python app/uc2_api.py
```

- `GET /health` says whether the optional model and the trained tagger are loaded.
- `POST /extract` takes `{"text": "...", "mode": "rules"}`, `"mode": "rules_model"` or `"mode": "rules_tagger"`. It returns the validated client profile: values, flags, the text each value came from, and whether each value came from a rule, the model or the tagger (`crf`).
- The interactive documentation is at http://127.0.0.1:8001/docs. The page loads its scripts from a public CDN, so it needs internet access.

Example (Git Bash, macOS or Linux):

```
curl -X POST http://127.0.0.1:8001/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "My wife and I want 10 nights in Kerala in March 2027, budget USD 9,000.", "mode": "rules"}'
```

If the model files or the tagger are missing, `rules_model` and `rules_tagger` fall back to rules only, and the response says so in `note`. Texts longer than 20,000 characters are refused.

Good English results do not show that the extractor works in French or German.

## Folder layout

```
data/uc1_hotel/       Hotel Booking Demand data (not committed)
data/uc2_inquiries/   Training data and test sets (not committed)
models/               Downloaded and trained models (not committed)
notebooks/            Exploration notebooks
src/uc1_rates/        Use Case 1 code
src/uc2_extractor/    Use Case 2 code
src/common/           Shared helpers
app/                  Streamlit demos
tests/                pytest tests
docs/                 Project documents
```

## Setup

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The datasets are not included in this repository. Download them yourself and place them in the `data/` folders listed above.

## Attribution

The trained tagger uses MASSIVE and Taskmaster-2, both under the Creative Commons Attribution 4.0 (CC BY 4.0) licence. Their attribution and the changes made to them are in `NOTICE`.

The licence of the Hotel Booking Demand dataset has not been checked, so this project makes no claim about it.
