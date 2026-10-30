# Kenexoft AI/ML Internship Project

This repository holds two separate AI/ML use cases built during an internship at Kenexoft, a luxury South Asia travel company.

## Use Case 1: Hotel Rate Prediction and Demand Analysis

Builds a reference hotel nightly rate from historical booking data and tests whether booking-demand signals make that reference rate more accurate. It uses the public Hotel Booking Demand dataset, which covers two hotels in Portugal from 2015 to 2017.

The recorded rate (adr) is not a live quote. Results on this dataset say nothing about how the model would perform on South Asian luxury hotels.

## Use Case 2: Multilingual Travel Inquiry Information Extractor

Pulls key details out of travel inquiries written in English, French or German: destination, dates or duration, travellers, budget, accommodation preferences, flight requirements and special requests. It shows the text that supports each value and flags information that is missing or unclear.

The first version is rule-based. It returns a validated JSON profile with the fields destination, dates or duration, travellers, budget (band plus original amount and currency), accommodation, flights, interests, dietary needs, occasion, special requests and language. Fields that are not found stay empty, and missing or ambiguous details are flagged.

It is evaluated on a small synthetic gold set written for this project (60 enquiries, 20 per language). Results and limits are in `docs/uc2_results.md`. The labelling rules are in `docs/uc2_labelling_guide.md`. The budget bands in `src/uc2_extractor/budget_bands.toml` are placeholders.

Try it:

```
python app/uc2_extract_cli.py --text "My wife and I want 2 weeks in Kerala in March, budget USD 20,000."
python -m src.uc2_extractor.evaluate
python -m pytest
```

Taskmaster-2 (CC BY 4.0, see Attribution below) was considered as a data source but was not used in the first version. Good English results do not show that the extractor works in French or German.

## Folder layout

```
data/uc1_hotel/       Hotel Booking Demand data (not committed)
data/uc2_inquiries/   Taskmaster-2 data and gold set (not committed)
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

Taskmaster-2 is released by Google under the Creative Commons Attribution 4.0 (CC BY 4.0) licence, as stated in its official README. It is not used in the current version of the extractor.

The licence of the Hotel Booking Demand dataset has not been checked, so this project makes no claim about it.
