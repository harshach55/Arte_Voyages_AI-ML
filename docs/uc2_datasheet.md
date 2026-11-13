# UC2 Training Data Datasheet

This describes the data used to train the slot tagger. The data is not stored in this repository.

## 1. Summary

The first tagger (v1) was trained on two public datasets, converted to our labels. The second tagger (v2) also uses 900 synthetic enquiries from our own generator (section 9). No client messages were used anywhere.

## 2. Sources

| | MASSIVE 1.0 | Taskmaster-2 |
|---|---|---|
| Licence | CC BY 4.0. The archive's LICENSE matches the CC BY 4.0 legal text word for word (ignoring whitespace), with one added line: "Copyright Amazon.com Inc. or its affiliates." | CC BY 4.0, according to its own README |
| Attribution | See `NOTICE` | See `NOTICE` |
| Languages used | English, French, German | English |
| Kind of text | Short voice commands to an assistant | Transcribed spoken dialogues |
| Files used | en-US, fr-FR and de-DE | hotels.json and flights.json, USER turns only |
| Raw size | Archive of 39,500,415 bytes | About 22.5 MB and 23.0 MB |
| Rows | 16,521 per language (11,514 train, 2,033 dev, 2,974 test) | 2,357 hotel and 2,481 flight dialogues |

The MASSIVE card's summary says 19,521 rows per language, but the data files hold 16,521. The Taskmaster-2 README says 2,355 hotel dialogues, but the file holds 2,357. The counts above are from the files.

## 3. After conversion

| | Examples kept | Dropped |
|---|---|---|
| MASSIVE | 2,309 (en 769, fr 774, de 766) | 32 exact repeats |
| Taskmaster-2 | 14,429 (12,527 with labels, the rest plain turns) | 2,385 exact repeats, 50 too close to test text |

Labelled spans:

| Label | English | French | German |
|---|---|---|---|
| DATE | 7,607 | 784 | 771 |
| DESTINATION | 4,457 | 0 | 0 |
| FLIGHT | 2,529 | 0 | 0 |
| ORIGIN | 1,627 | 0 | 0 |
| ACCOMMODATION | 921 | 0 | 0 |
| PARTY | 522 | 0 | 0 |
| ADULTS | 203 | 0 | 0 |
| DURATION | 159 | 0 | 0 |
| CHILDREN | 59 | 0 | 0 |
| AGE, BUDGET, INTEREST, DIETARY, OCCASION, SPECIAL_REQUEST | 0 | 0 | 0 |

Plain turns without any annotation are kept at 15% of the labelled count, so the tagger also sees text with nothing in it.

## 4. Training and development split

| | Train | Development |
|---|---|---|
| Examples | 13,965 | 2,773 |
| MASSIVE | 1,641 | 668 |
| Taskmaster-2 | 12,324 | 2,105 |
| English | 12,862 | 2,336 |
| French | 557 | 217 |
| German | 546 | 220 |

- MASSIVE uses its own partitions: train for training, dev and test for development. A sentence and its translations share a partition, so a translation cannot cross the split.
- Taskmaster-2 is split by whole dialogue, using a hash of the dialogue id (15% for development).
- The development split comes from the same public sources as the training data. It is easier than real enquiries.

## 5. Keeping test data out

Every converted example was checked against the gold set (60 texts), held-out set 1 (45 texts) and the corrections file (1 text). Held-out set 2 has no enquiries yet, so it was not part of this check. When it arrives, this check must be run again.

The check holds hashes of the test text, not the text itself. After normalising (lower case, no accents, no punctuation), an example is dropped if:

- the whole text equals a test text
- it shares a run of 6 or more words with a test text
- 50% or more of the shorter text's 3-word groups appear in a test text
- its 5-letter groups overlap a test text by 60% or more
- a sentence of 6 or more words overlaps a test sentence by 50% or more (2-word groups)

Line 1 of the corrections file is an unedited copy of gold item P02-en and is excluded from all training data (see `NOT_FOR_TRAINING.txt` beside it).

## 6. File fingerprints (SHA-256)

| File | SHA-256 |
|---|---|
| massive/amazon-massive-dataset-1.0.tar.gz | 7df623fd2d300a4d235d6ee5bd396c9a28258d3a0ccb29abdb054506eba153f8 |
| massive/extracted/1.0/LICENSE | c2e6ea015269147de02117ebdd91f30ef09831251f5345fa8365273b1db1d435 |
| massive/extracted/1.0/data/en-US.jsonl | c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e |
| massive/extracted/1.0/data/fr-FR.jsonl | f9bf3db170ad415b389e4c9594dd0f8f80c38188143e05cc4459a6fa7df7cf49 |
| massive/extracted/1.0/data/de-DE.jsonl | 5e09cc550b38d37faf002e0ff42103acd330906e23032689e8f071e1cf3ff621 |
| taskmaster2/data/hotels.json | 975b0242f1e37ea1ab94ccedd7e0d6ee5831599d5df1f16143e71110d6c6006a |
| taskmaster2/data/flights.json | 86b37b5ae25f530fd18ced78800d30c3b54f7b34bb208ecb51842718f04e760b |
| taskmaster2/README.md | 0e707d4cc31869e0c0337979f72ecef61fc4fcab2bfe3a4ec1a546b84bc17463 |
| taskmaster2/ontology/hotels.json | 22ae51ba546ee7ca03143097782817c4cdd0de74ac84893eaf40b8254aa866d3 |
| taskmaster2/ontology/flights.json | 1ebc5c982339d24b2dcf50677883fed65b7fcb95f01edbbd3be6357090893c33 |

Fingerprints of the converted files are in `models/uc2_crf/crf_tagger.json`.

## 7. Known gaps

- Neither source looks like a client email. MASSIVE is short voice commands and Taskmaster-2 is transcribed speech, with some speech recognition errors.
- French and German have about 770 short date examples each, and nothing else.
- Six of our 15 labels have no training data at all.
- Neither dataset was collected for luxury travel to South Asia.
- A trained model can only learn the wording in these sources, so long client emails are a poor match for it.

## 8. Not checked yet

- How CC BY 4.0 applies to weights trained on the data.
- How much speech recognition noise Taskmaster-2 has; it was seen but not measured.
- Why the MASSIVE card's row count differs from the data files.
- Near-duplicates between the training and development parts of the same source were not removed, apart from exact repeats.

## 9. Synthetic enquiries (pilot, used by the second tagger)

- **What:** 900 enquiries, 300 per language (English, French and German), written by the generator in `src/uc2_extractor/training/synth/`. Seed 20261106, so running it again gives the same data.
- **Labels:** known by construction. Every value is placed into the text by the generator, so its position is exact. Every example is checked the same way as the public data (spans match the text, BIO round trip).
- **Fields covered:** destination, dates, trip length, party, adults, children, ages, budget, accommodation, flights, origin, interests, dietary needs, occasion and special requests.
- **Styles:** short notes (like a quick message), medium web-form messages and long formal emails. Fields come in a random order, some have typos, numbers are written as words or digits, and some details are missing. Texts also contain things that look like a field but are not: past trips, home towns, other dates and other amounts.
- **No real people or real text:** names are invented, and no sentence was copied from a test set or a client message.
- **Kept away from test text:** each enquiry was checked against the gold set, held-out set 1 and the corrections with the same hash rules as in section 5, and 14 drafts were rejected. The wording was written by someone who had read the gold set and held-out set 1, so those two sets are no longer blind for anything trained on this data. Only held-out set 2 is blind.
- **Split:** by a hash of the enquiry id: 720 training, 79 development, 101 test (test: en 43, fr 30, de 28). The test part is never used for training or tuning.
- **Not checked by a native speaker:** the French and German text has not been checked by a native speaker yet.
- **Limits:** the phrases come from fixed word lists, so the variety is smaller than in real email. Scores on the synthetic test split are optimistic for that reason.
