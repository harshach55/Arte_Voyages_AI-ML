# UC 2.1 Hotel Rate Prediction - First Version

Date: 13 November 2026

## 1. What this is

A first version of a model that gives a reference nightly rate (in EUR) for a hotel booking,
plus a 10-90% price range. It is trained on public data from two hotels in Portugal. The aim is
to prove the method and the pipeline. It is not a model of South Asian luxury hotels.

Code: `src/uc21_rates/`. Saved models and results: `models/uc21_rates/` (not in git).
Data: `data/uc21_rates/raw/` (not in git).

## 2. Data source and licence

**Paper:** Nuno Antonio, Ana de Almeida, Luis Nunes. "Hotel booking demand datasets".
Data in Brief, volume 22, 2019, pages 41-49. doi: 10.1016/j.dib.2018.11.126

**Licence sentence (copied word for word from the article):**

> "This is an open access article under the CC BY license (http://creativecommons.org/licenses/by/4.0/)."

The copyright line in the same place is "© 2018 The Authors".

**Where I read it:** the article's full text from PubMed Central (PMC6297060), downloaded from
Europe PMC: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6297060/fullTextXML
(the ScienceDirect page and the PMC web page blocked automatic access, so they could not be read).
The Crossref record for the DOI (https://api.crossref.org/works/10.1016/j.dib.2018.11.126) also lists
http://creativecommons.org/licenses/by/4.0/ as the licence of the published version.

**Does it cover the data?** The paper's specifications table says:

| Row | Text in the paper |
|---|---|
| Data accessibility | "Data is supplied with the paper" |
| Data source location | "Both hotels are located in Portugal: H1 at the resort region of Algarve and H2 at the city of Lisbon" |

The data files are published as supplementary material of the same article (mmc2.zip, mmc3.zip and
mmc4.zip under "Appendix A. Supplementary material"). So the data is part of a CC BY 4.0 article.
Note: the licence sentence is written for the article and does not list the supplementary files
one by one.

**Attribution:** CC BY 4.0 needs credit. Anyone who uses this data or results must cite the paper above.

**The copy we used.** We did not download the publisher's own files. We used the TidyTuesday copy
(2020-02-11): https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv
Its README says the data "comes from an open hotel booking demand dataset from Antonio, Almeida and
Nunes, 2019". Its cleaning script reads H1.csv (labelled "Resort Hotel") and H2.csv (labelled
"City Hotel"), cleans the column names and stacks the two files. The README has no licence statement
of its own.

| File fact | Value |
|---|---|
| File | data/uc21_rates/raw/hotels.csv |
| Downloaded | 13 November 2026 |
| Size | 16,855,599 bytes |
| Rows | 119,390 (plus 1 header line), 32 columns |
| SHA-256 | 7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06 |

## 3. Cleaning

The target is `adr` (average daily rate, EUR). The paper defines it as the sum of all lodging
transactions divided by the number of staying nights. Filters, in this order:

| Filter | Removed | In train period | In test period | Rows left | Why |
|---|---|---|---|---|---|
| Raw file | - | - | - | 119,390 | |
| Cancelled bookings (is_canceled = 1) | 44,224 | 28,479 | 15,745 | 75,166 | the stay never happened |
| adr of 0 or less | 1,747 | 1,343 | 404 | 73,419 | free or corrected stays, not a paid rate |
| Zero nights | 0 | 0 | 0 | 73,419 | no night, so no nightly rate (none left after the first two filters) |
| Outlier adr | 139 | 101 | 38 | 73,280 | see below |

**Outlier rule (decided from the training period only):** the median adr of the training rows
(2015-2016, after the first three filters) is 89.75 EUR. A rate below one tenth of it (8.975 EUR) or
above ten times it (897.50 EUR) is treated as a data error. All 139 removed rows were at the low end;
no row was above 897.50 EUR. (The well-known 5,400 EUR row is a cancelled booking and was already
removed by the first filter.) To be open about it: before this rule was set, the highest adr values
in the 2017 rows were printed once. The rule itself only uses the training median.

Other small steps (no rows removed):
- `children` had 4 missing values; they are set to 0.
- `meal` value "Undefined" is merged into "SC". The paper defines them as one category ("Undefined/SC", no meal package).

**Two things found in the data file (checked on all 119,390 rows, 100% match):**
- `stays_in_weekend_nights` counts the nights that start on a **Sunday or a Monday**. The paper describes
  them as Saturday or Sunday nights. We do not know why. `predict.py` uses the rule that matches the file,
  so a new booking gets the same kind of input the model was trained on.
- `arrival_date_week_number` uses weeks that start on Sunday, with week 1 being the week of 1 January.
  `predict.py` uses the same rule.

## 4. Features

**Used (all known when the booking is made):** hotel, arrival month, arrival week number, arrival
day of week (from year/month/day), lead_time, stays_in_weekend_nights, stays_in_week_nights, adults,
children, babies, meal, market_segment, distribution_channel, reserved_room_type, customer_type,
is_repeated_guest, required_car_parking_spaces, total_of_special_requests.

**Excluded, because they are only known after booking or are the outcome (leakage):**
assigned_room_type, reservation_status, reservation_status_date, booking_changes,
days_in_waiting_list, is_canceled.

**Excluded, because they are not in the agreed list for this version:** country, agent, company,
deposit_type, previous_cancellations, previous_bookings_not_canceled.

The year is not a feature. So the model cannot follow price changes between years (see section 7).

## 5. Time split and test set lock

| Part | Arrivals | Rows |
|---|---|---|
| Training | 2015-07-01 to 2016-12-31 | 48,780 |
| Test (locked) | 2017-01-01 to 2017-08-31 | 24,500 |

There is no random split. The data ends on 31 August 2017, so the test set covers January to August only.

**Choices (validation slice inside training):** the models were fitted on arrivals up to 2016-08-31
(36,764 rows) and checked on 2016-09-01 to 2016-12-31 (12,016 rows). Only this slice was used to pick:
- Ridge alpha: 1000 (tried 0.1, 1, 10, 100, 1000, 3000, 10000; validation MAE 23.42 EUR)
- Gradient boosting: learning rate 0.05, 15 leaves, 100 iterations (24 settings tried; validation MAE 18.96 EUR)
- The Ridge and boosting grids were widened twice, because the best value sat at the edge of the grid.
  This used the validation slice only, before the test set was scored.
- The two quantile models (10% and 90%) use the same settings. Their range covered 72.8% of the validation rows.

After the choices, all models were refitted on the full training period.

**Test set fingerprint:** before scoring, the cleaned test rows (sorted by row number in the raw file,
fixed column order, written as CSV) were hashed with SHA-256 and saved in `models/uc21_rates/test_lock.json`:

- rows: 24,500
- SHA-256: f75f626d137ea8df17f1b41803ec41966889347e81e54850eea6868dbe05e044
- locked on: 2026-11-13 17:57:33

The test set was scored once, on 2026-11-13 at 17:57:39. Nothing was changed after seeing the results.
`train.py` refuses to score again while `models/uc21_rates/metrics.json` exists.

## 6. Results on the locked 2017 test set

MAE = mean absolute error in EUR (lower is better). MAPE = mean absolute percentage error.
Coverage = share of test rows where low <= actual rate <= high.

**Both hotels (24,500 rows)**

| Model | MAE (EUR) | MAPE (%) |
|---|---|---|
| Baseline (median per hotel + month + room type) | 24.76 | 21.3 |
| Ridge | 25.15 | 22.0 |
| Gradient boosting | 21.14 | 17.9 |

10-90% range (two quantile gradient boosting models): coverage 63.5%, mean width 52.80 EUR.
The demo shows the range widened where needed so it contains the predicted rate (this was needed for
99 of the 24,500 rows). That changes the coverage only very slightly: 63.490% raw, 63.494% as shown
(metrics.json). The 10% and 90% predictions never crossed.

**Per hotel**

| Hotel | Rows | Baseline MAE | Ridge MAE | Boosting MAE | Baseline MAPE | Ridge MAPE | Boosting MAPE | Range coverage |
|---|---|---|---|---|---|---|---|---|
| City Hotel | 15,555 | 24.09 | 23.02 | 21.21 | 19.8 | 19.8 | 17.7 | 59.6% |
| Resort Hotel | 8,945 | 25.94 | 28.87 | 21.01 | 23.8 | 25.9 | 18.4 | 70.3% |

Mean width of the range: City Hotel 49.60 EUR, Resort Hotel 58.36 EUR.

**Top 8 drivers of the gradient boosting model** (permutation importance on a sample of 5,000 test
rows, 5 repeats, random_state 0; the number is how much the MAE rises, in EUR, when that column is shuffled):

| Rank | Feature | MAE increase (EUR) |
|---|---|---|
| 1 | arrival_week | 10.30 |
| 2 | hotel | 6.98 |
| 3 | reserved_room_type | 4.96 |
| 4 | market_segment | 3.54 |
| 5 | meal | 2.12 |
| 6 | adults | 1.87 |
| 7 | arrival_month | 1.24 |
| 8 | children | 0.91 |

distribution_channel came last with -0.08 (shuffling it made the error slightly smaller), so the
channel input adds nothing on this test set.

## 7. What the results mean

- Gradient boosting has the lowest error: MAE 21.14 EUR against 24.76 EUR for the baseline, about 15% lower.
- Ridge is worse than the baseline overall (25.15 EUR). It is a little better at the City Hotel but much worse at the Resort Hotel.
- The 10-90% range should hold about 80% of real rates if it were calibrated. It held 63.5%
  (59.6% City, 70.3% Resort). So it is too narrow and must not be read as a guarantee.
- Errors on 2017 are higher than on the 2016 validation slice (21.14 against 18.96 EUR). A likely reason:
  rates in 2017 were higher than in the same months of 2016 (table below), and the model has no year
  input, so it cannot follow that rise. We did not test this further.

| Arrivals January-August | City Hotel mean adr | City Hotel median | Resort Hotel mean adr | Resort Hotel median |
|---|---|---|---|---|
| 2016 (training) | 104.87 | 99.00 | 92.14 | 71.43 |
| 2017 (test) | 118.78 | 112.67 | 103.80 | 82.00 |

(Cleaned rows. These are plain averages of the data, not model results.)

## 8. Model card (short)

- **What it does:** gives a reference nightly rate in EUR and a 10-90% range for a booking at one of two hotels.
- **Data:** European data from two Portuguese hotels (a resort in the Algarve and a city hotel in Lisbon), arrivals July 2015 to August 2017. Not South Asian hotels, and not luxury hotels in particular.
- **Purpose of this version:** to prove the method and the pipeline (time split, locked test set, baseline, error measures, demo). A real model for Kenexoft needs Kenexoft's own rate data.
- **adr is an average daily rate** over the whole stay, as recorded by the hotel. It is not a live quote and not a price offer.
- **The 10-90% range is not a guarantee.** The raw range held 63.5% of actual 2017 rates; after conformal calibration it held 71.0%, still below the 80% target (section 10).
- **Price level:** there is no year input, so predictions are at 2015-2016 price levels.
- **Room types** are letters only; the paper says codes are used instead of names for anonymity.
- **Missing inputs** (market segment, customer type, repeat guest, parking, special requests, babies) are filled with the most common training value: Online TA, Transient, 0, 0, 0, 0.

## 9. How to run

```
.venv\Scripts\activate
python -m src.uc21_rates.train --validate-only    (choices only, test set not touched)
python -m src.uc21_rates.train                    (full run; refuses if metrics.json exists)
python -m pytest tests/test_uc21_rates.py -q
python app/uc21_rate_demo.py                      (then open http://localhost:8003)
```

The printed output of the one real run is saved in `models/uc21_rates/train_output.txt`.

**Demo page** (`app/uc21_rate_demo.py`, port 8003, this computer only): a form for one booking
(hotel, arrival date, nights, adults, children, room type, meal, channel, lead time). It shows the
predicted nightly rate and the 10-90% range, a chart of the same booking in each month of 2017, the
locked test set results (read from `models/uc21_rates/metrics.json` when the page starts) and the top
drivers. Small "i" icons explain each part.

Use in code:

```
from src.uc21_rates.predict import predict_rate, seasonality, allowed_values
predict_rate({"hotel": "City Hotel", "arrival_date": "2017-05-12", "nights": 3, "adults": 2,
              "children": 0, "room_type": "A", "meal": "BB", "channel": "TA/TO", "lead_time": 30})
```

Software: Python 3.12.4, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4.

## 10. Calibrated 10-90% range (conformalized quantile regression)

The raw range was too narrow (63.5% instead of 80%), so it was calibrated with conformalized
quantile regression. All choices were fixed before the 2017 test set was used again:

- Calibration slice: arrivals 2016-09-01 to 2016-12-31 (12,016 rows).
- The 10% and 90% quantile models were refitted on the rest of 2015-2016 (36,764 rows, arrivals
  2015-07-01 to 2016-08-31), with the same settings as before.
- For each calibration booking: E = max(low - actual, actual - high). Q is the ceil((n + 1) x 0.8)-th
  smallest E. Here Q = 4.38 EUR, so each end of the range moves out by 4.38 EUR.
- On the calibration slice itself, the raw range held 72.8% and the calibrated range
  80.0% (80% by construction, so this is not a test).
- Note: the boosting settings were chosen earlier on this same slice for the point rate (by MAE, not by
  coverage).

`calibrate.py` allows exactly one scoring of the locked 2017 set for the calibrated range (it checks the
fingerprint and refuses a second run). It was scored once, on 2026-11-13 at 18:38:30:

| Range on the locked 2017 test set | Coverage | Mean width (EUR) |
|---|---|---|
| Raw quantile models (from section 6) | 63.5% | 52.80 |
| Calibrated, both hotels | 71.0% | 61.04 |
| Calibrated, City Hotel (15,555 rows) | 67.0% | 58.00 |
| Calibrated, Resort Hotel (8,945 rows) | 77.9% | 66.32 |

The calibrated range is better but still below 80% on 2017. The method only promises its coverage when new
bookings behave like the calibration bookings, and 2017 prices were higher than 2016 prices (section 7).
The demo and the API now show the calibrated range. The point rate and the MAE/MAPE results are unchanged.
