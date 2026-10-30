# UC2 Gold Set: Labelling Guide

This guide explains how the gold set for the travel enquiry extractor is stored and labelled. Both annotators should follow it so that their labels can be compared.

## 1. Where the files are

The gold set is kept in `data/uc2_inquiries/gold/`. This folder is not committed to git.

- `enquiries.csv` has one row per enquiry: `id`, `group_id`, `language`, `text`, `review_status`, `review_note`, `second_annotator`.
- `labels.csv` has one row per enquiry and field: `id`, `field`, `gold_value`, `annotator_2_value`, `annotator_2_notes`.

Both files are UTF-8 with a BOM, so Excel shows accents correctly. Every enquiry has a row for every field, including fields that are empty.

All enquiries are synthetic. They were written for this project and are not real customer messages.

## 2. How the set is built

- 60 enquiries: 20 English, 20 French, 20 German.
- 36 of them are 12 paired groups (P01 to P12). Each group is the same enquiry written in all three languages.
- 24 are single enquiries (E13-E20, F13-F20, D13-D20) with different styles: informal, long emails with signatures, very short messages, and messages with missing or unclear details.

## 3. Review status (what you need to do)

| review_status | Meaning |
|---|---|
| rewrite | Too close to the extraction rules, so it may make the scores look too good. Please rewrite it in your own words and relabel it. |
| check language | French or German text written by a non-native writer. A fluent speaker should check that it sounds natural. |
| check labels | One or more labels are a judgement call. The `review_note` says which. |
| check language + labels | Both of the above. |
| ok (spot check) | No known issue. Please spot-check it anyway. |

The `second_annotator` column marks 12 enquiries (20 percent). A second person should label these on their own, without looking at `gold_value`. They write their answers in `annotator_2_value`. The evaluation script then reports exact agreement.

## 4. Value formats

- Lists use ` | ` between items, for example `India | Nepal`. Order does not matter.
- Dates: `2027-03-04` (full date), `--03-04` (no year written), `2027-03` or `--03` (month only).
- Duration: number, a space, then `nights`, `days` or `weeks`, for example `14 nights` or `1 weeks`. Write the unit the client used. Do not convert.
- Numbers: plain digits with no separators, for example `30000`.
- Currency: ISO code (`USD`, `EUR`, `GBP`, `INR`, `CHF`).
- Labels for accommodation, flights, interests, dietary needs, occasion and special requests come from the fixed list in `src/uc2_extractor/lexicon.py`. They are written in English whatever the language of the enquiry.
- Destination: English place names. A real place that is not in the list (for example Pangong Lake) can still be written down.

## 5. Labelling rules

**General**
- Label only what the text says. If it is not written, leave it empty.
- Do not label things the client says they do not want ("no shopping", "keine Inlandsflüge").

**Dates**
- `start_date` and `end_date` are filled only when the day and month are clear. If only the month is clear, use the month form.
- A numeric date where both numbers could be the month (3/4, 02.01.) goes in `dates_ambiguous`, written exactly as in the text, and the start and end stay empty. This applies even when the context makes one reading likely.
- Never add a year that is not written.
- Holidays and seasons (Easter, Toussaint, next autumn) are not dates.
- Alternatives ("late October or early November") are not a range, so leave the dates empty.
- Dates that are not travel dates are ignored, for example the date of a phone call or an email signature.

**Duration**
- Record the length of the whole trip. If only parts are given ("4 nights in Kathmandu, 3 nights in Chitwan") with no total, leave it empty.
- A range ("7-8 nights") is left empty.

**Travellers**
- `party_size` is the total number of people travelling. Use the stated total, or adults plus children when both are known.
- `adults` is filled when it is clear the travellers are adults ("my husband and I", "three friends", "4 Erwachsene"). A daughter aged 18 counts as an adult.
- `children` counts people under 18.
- `ages` are ages written for travellers. A baby given in months is age 0 (please confirm this rule). An age only implied by a birthday ("turns 60", "fête ses 60 ans") is not labelled as an age.

**Budget**
- `budget_min` and `budget_max` are the amounts written. "Up to X" fills only the maximum, "at least X" only the minimum, and a single amount fills both.
- A bare "$" or "dollars" is labelled USD (the extractor also flags this as an assumption).
- `budget_basis` is `per_trip` or `per_person`, and only when the text says so.
- `budget_band` is filled only for USD budgets that are not per person, using the PLACEHOLDER bands in `src/uc2_extractor/budget_bands.toml`. If a range covers two bands, leave the band empty. If only a maximum is given, use the maximum.
- Currencies are never converted.

**Other fields**
- Interests are activities or themes the client wants. Hotel features (beachfront) belong in accommodation, not in interests.
- A special request or must-have is labelled only if it matches a label in the list. Wording that clearly means a listed label also counts ("someone to show us around privately" means private guide).
