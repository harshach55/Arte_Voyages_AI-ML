# UC2 Label Scheme

This note describes the labels for the slot tagger and how the public datasets map onto them.

## 1. Format

- Each training example keeps its labels as character spans, so a label always points at exact text.
- BIO tags come from those spans, using the project tokenizer in `src/uc2_extractor/text.py`.
- There are 15 labels, so 31 tags: `O`, plus `B-` and `I-` for each label.
- Every example is checked: each span must match the text, sit on token boundaries, and survive the trip from spans to BIO and back.
- Text that a source annotated with something we do not use is marked "ignore" (tag `IGN`). The tagger is not trained on those tokens, so it is not told they are plain text.

## 2. Labels

| Label | Meaning | Example | Training data |
|---|---|---|---|
| DESTINATION | A place the client wants to visit | Kerala | English |
| ORIGIN | The city or airport they would fly from | London | English |
| DATE | A travel date, month or date range, as written | 3 to 12 March | English, French, German |
| DURATION | The length of the whole trip | ten nights | English |
| PARTY | The total number of travellers | four people | English |
| ADULTS | The number of adults | 2 adults | English |
| CHILDREN | The number of children | three kids | English |
| AGE | Ages of travellers, numbers only | 7 and 10 | None yet |
| BUDGET | Money for the trip, with qualifier, currency and basis | around €20,000 per person | None yet |
| ACCOMMODATION | The kind of place to stay | five-star hotel | English, star ratings only |
| FLIGHT | Cabin, direct, domestic, or no flights needed | business class | English |
| INTEREST | What they want to do or see | tiger safaris | None yet |
| DIETARY | Food needs | gluten-free | None yet |
| OCCASION | What the trip celebrates | honeymoon | None yet |
| SPECIAL_REQUEST | Services and must-haves | private guide | None yet |

## 3. Labelling rules

- DATE does not include a leading preposition: in "from 3 March", the span is "3 March".
- DATE has no start or end role. The date parser decides that later.
- "February or March" is one DATE span, so the date parser can flag the choice.
- DURATION is the length of the whole trip. "Three nights in Jaipur" as one stop stays plain text.
- BUDGET includes the qualifier, amount, currency and basis when they sit together.
- "My wife and I" is ADULTS. "The two of us" is PARTY.
- ORIGIN is only for flights. The client's home town stays plain text.
- Things the client refuses, and past trips, stay plain text.
- Dates, money and people that are not about this trip stay plain text.

## 4. Mapping from MASSIVE

The slot names were read from the `[slot : text]` tags in the en-US, fr-FR and de-DE data files. There are 55 slot names, the same in all three languages. Only one maps cleanly.

| MASSIVE slot | Our label | Notes |
|---|---|---|
| date | DATE | Only the calendar, transport, recommendation and weather scenarios. Past dates, clock times and stray words removed. Each value used at most 25 times per language. |
| place_name | Ignored | It never says whether a place is a destination ("nach hause", "gare", weather places). |
| time, timeofday | Ignored | Not a travel date. |
| All others | Not used | For example food_type, event_name, transport_type and currency_name do not fit our labels cleanly. |

## 5. Mapping from Taskmaster-2

Only USER turns from the hotels and flights files are used.

| Taskmaster-2 annotation | Our label | Notes |
|---|---|---|
| hotel_search.location.hotel | DESTINATION | |
| flight_search.destination1, destination2 | DESTINATION | |
| flight_search.origin | ORIGIN | |
| check-in date, check-out date | DATE | Not when the text is about the act of leaving or arriving |
| date.depart_origin, date.depart_intermediate | DATE | |
| date.return | DATE | Not relative returns such as "in 4 days" |
| date_range | DATE or DURATION | DURATION when it is purely a length, such as "two nights" |
| num.guests, num.pax | PARTY, ADULTS or CHILDREN | Only with a person noun. A bare "two" is ignored because it needs the question before it. |
| star_rating | ACCOMMODATION | Only with star wording |
| seating_class | FLIGHT | Strict pattern for cabin names |
| stops | FLIGHT | Only nonstop or direct wording |
| Everything else | Ignored | price_range, amenity, type.room, other_request, sub_location, time_of_day, airline and others |

Price ranges are ignored because they are nightly or ticket prices, not a trip budget.

## 6. Gaps

- AGE, BUDGET, INTEREST, DIETARY, OCCASION and SPECIAL_REQUEST have no training data. The tagger cannot find them.
- French and German have DATE only.
- ACCOMMODATION comes from star ratings only, and the accommodation field has one star label (5-star), so the tagger rarely adds anything there.
