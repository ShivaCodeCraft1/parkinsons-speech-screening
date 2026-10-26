# Dataset verification report

**Overall hard-fail verdict: FAIL**  
Run (UTC): 2026-10-22T11:25:19+00:00 | script v1.1.0 | git commit: not available

> Every number below was measured from the local files on this run. Nothing is taken from papers. Participant names are not included.

## 1. Hard-fail checks

| Check | Status | Affected | Details |
|---|---|---|---|
| H0_wav_files_found | PASS | 0 | 831 WAV files found |
| H1_every_wav_has_exactly_one_label | PASS | 0 | label status counts: {'ok': 831}. Fix group regexes in the config if wrong. |
| H2_subject_identity_unambiguous | FAIL | 1 | WAVs not inside a participant folder: 0; subject_id hash collisions: 0; participant names repeated across groups: 1; different folders with the same normalised name in one group: 0 |
| H3_subject_leakage_risk | FAIL | 1 | same name in different groups: 1; identical audio shared by different subjects: 0. Cannot detect one person stored under two differently named folders -> verify by hand. |
| H4_all_audio_readable_and_non_empty | PASS | 0 | 0 unreadable or empty WAVs |
| H5_task_codes_classified_or_quarantined | PASS | 0 | task parse status counts: {'ok': 766, 'unknown_code': 65}; unknown_code rows are written to the exclusion manifest |
| H6_no_duplicate_recording_content | FAIL | 2 | byte-identical groups: 1; identical-audio groups: 1; subject+task slots with >1 file (review only): 45 |
| H7_recording_ids_unique | PASS | 0 | 0 repeated recording_id values |

## 2. Warnings

- A new salt file was created in the private folder. Back it up: without it subject_id values change.
- Metadata join status is PARTIAL: age/sex are left EMPTY for participants that could not be linked unambiguously (nothing is guessed).
- 65 recordings have unknown task codes and were preserved in the full manifest but written to the exclusion manifest; no codes were guessed.
- 45 subject+task slots contain multiple recordings. These are retained and listed for review; only identical content triggers H6.
- Other file types present (ignored): {'.txt': 1}
- Participant-folder count for 'parkinsons_disease' is 24 but you counted 28 by hand.
- More than one sample rate present: {16000: 671, 44100: 160} (resampling policy is a Phase 8 decision).
- QC flag 'clipping_suspected' on 11 files (heuristic; inspect before Phase 6 decisions).
- SHORTCUT RISK: 'sample_rate' values differ between label groups ({'young_healthy_control': ['16000'], 'elderly_healthy_control': ['16000'], 'parkinsons_disease': ['16000', '44100']}). A model could learn the recording setup.

## 3. Counts

- WAV files found: **831**
- Participant folders found (structure): **61**
- Subjects with at least one WAV: **61**
- Other audio files (unexpected formats): 0
- Excel files: 3 | FILE CODES files: 3
- Other file extensions present: {'.txt': 1}

| Label | Participant folders | Subjects with audio | Recordings | Your manual count |
|---|---|---|---|---|
| young_healthy_control | 15 | 15 | 45 | 15 |
| elderly_healthy_control | 22 | 22 | 349 | 22 |
| parkinsons_disease | 24 | 24 | 437 | 28 |

## 4. Recordings per subject

| Group | n subjects | min | median | mean | max |
|---|---|---|---|---|---|
| young_healthy_control | 15 | 3 | 3 | 3 | 3 |
| elderly_healthy_control | 22 | 13 | 16 | 15.86 | 16 |
| parkinsons_disease | 24 | 10 | 16 | 18.21 | 48 |

Histogram of recordings-per-subject (all groups): {3: 15, 10: 1, 13: 1, 14: 2, 15: 1, 16: 38, 32: 2, 48: 1}

## 5. Recordings per task

| Task code | Task name | young_healthy_control | elderly_healthy_control | parkinsons_disease | Total | Subjects having it |
|---|---|---|---|---|---|---|
| B1 | balanced_text_reading_1 | 15 | 21 | 28 | 64 | 60 |
| B2 | balanced_text_reading_2 | 15 | 21 | 24 | 60 | 56 |
| D1 | syllable_task_5s_1 | 0 | 22 | 28 | 50 | 46 |
| D2 | syllable_task_5s_2 | 0 | 22 | 28 | 50 | 46 |
| FB1 | balanced_phrases_words_1 | 0 | 21 | 26 | 47 | 43 |
| FB2 | balanced_phrases_words_2 | 0 | 0 | 0 | 0 | 0 |
| VA1 | sustained_vowel_a_1 | 0 | 22 | 28 | 50 | 46 |
| VA2 | sustained_vowel_a_2 | 0 | 22 | 27 | 49 | 45 |
| VE1 | sustained_vowel_e_1 | 0 | 22 | 28 | 50 | 46 |
| VE2 | sustained_vowel_e_2 | 0 | 22 | 27 | 49 | 45 |
| VI1 | sustained_vowel_i_1 | 0 | 22 | 28 | 50 | 46 |
| VI2 | sustained_vowel_i_2 | 0 | 22 | 27 | 49 | 45 |
| VO1 | sustained_vowel_o_1 | 0 | 22 | 28 | 50 | 46 |
| VO2 | sustained_vowel_o_2 | 0 | 22 | 27 | 49 | 45 |
| VU1 | sustained_vowel_u_1 | 0 | 22 | 28 | 50 | 46 |
| VU2 | sustained_vowel_u_2 | 0 | 22 | 27 | 49 | 45 |

Unparsed / unknown task-code files: **65** (status counts: {'ok': 766, 'unknown_code': 65})

Subjects by number of distinct known task codes covered (of 16): {2: 15, 9: 1, 12: 1, 13: 2, 14: 1, 15: 41}

## 6. Filename shapes (names removed: digits->9, letters->A)

| Shape | Files |
|---|---|
| A9A9A9 | 786 |
| A9A A9A9 | 28 |
| A9A_A9A9 | 16 |
| A9A9A9. | 1 |

## 7. FILE CODES cross-check

- FILE CODES files found: 3
- format `.xlsx`: parsed=True, codes found=['B1', 'B2', 'D1', 'D2', 'FB1', 'FB2', 'VA1', 'VA2', 'VE1', 'VE2', 'VI1', 'VI2', 'VO1', 'VO2', 'VU1', 'VU2'], error=None
- format `.xlsx`: parsed=True, codes found=['B1', 'B2', 'D1', 'D2', 'FB1', 'FB2', 'VA1', 'VA2', 'VE1', 'VE2', 'VI1', 'VI2', 'VO1', 'VO2', 'VU1', 'VU2'], error=None
- format `.xlsx`: parsed=True, codes found=['B1', 'B2', 'D1', 'D2', 'FB1', 'FB2', 'VA1', 'VA2', 'VE1', 'VE2', 'VI1', 'VI2', 'VO1', 'VO2', 'VU1', 'VU2'], error=None
- Codes in FILE CODES but not in config: []
- Codes in config but not in FILE CODES: []

## 8. Audio inspection (all WAV files)

- Readable: 831 | unreadable: 0 | empty: 0
- Sample rates (Hz): {'16000': 671, '44100': 160}
- Channels: {'1': 831}
- Encodings (subtype): {'PCM_16': 831}
- QC flags (heuristic, thresholds in config): {'clipping_suspected': 11}
- Duration (s) overall: {'n': 831, 'min': 3.375, 'median': 11.57, 'mean': 24.04372149757416, 'max': 250.3125}

| Task | n | min s | median s | mean s | max s |
|---|---|---|---|---|---|
| B1 | 64 | 38.3 | 65.13 | 76.37 | 250.3 |
| B2 | 60 | 39.18 | 55.76 | 61.09 | 191.1 |
| D1 | 50 | 5.001 | 6.483 | 6.269 | 7.349 |
| D2 | 50 | 5.001 | 6.213 | 6.194 | 7.688 |
| FB1 | 47 | 23.87 | 30.3 | 34.01 | 75.44 |
| FB2 | 0 | n/a | n/a | n/a | n/a |
| VA1 | 50 | 4.375 | 12.35 | 13.84 | 27.6 |
| VA2 | 49 | 5.375 | 6.037 | 7.281 | 22 |
| VE1 | 50 | 4.375 | 12.41 | 14 | 32.25 |
| VE2 | 49 | 3.75 | 6.125 | 7.021 | 19.75 |
| VI1 | 50 | 4 | 13.71 | 14.46 | 29.13 |
| VI2 | 49 | 5.25 | 6.165 | 7.34 | 35.22 |
| VO1 | 50 | 6.5 | 13.23 | 14.13 | 31.6 |
| VO2 | 49 | 5.376 | 6.25 | 7.283 | 23.5 |
| VU1 | 50 | 3.375 | 12.85 | 14.14 | 29.03 |
| VU2 | 49 | 5.125 | 6.375 | 7.28 | 22.7 |

## 9. Duplicates and leakage proxies

- Byte-identical file groups: 1 (files involved: 2)
- Identical decoded-audio groups: 1 (files involved: 2)
- Duplicate-content groups spanning more than one subject: 0
- Subject+task slots with more than one file: 45

## 10. Excel metadata

- Files: 3

### `excel_file_1:15 YHC.xlsx` (readable: True, error: None)

**Sheet `Foglio1`** -- 17 rows x 11 columns (fully-empty rows: 0, redacted headers: 0)

| # | Column | Missing | Unique | Summary (only non-sensitive) |
|---|---|---|---|---|
| 0 | Unnamed: 0 | 1 | 14 | values hidden |
| 1 | Unnamed: 1 | 1 | 6 | values hidden |
| 2 | Unnamed: 2 | 1 | 3 | values hidden |
| 3 | Unnamed: 3 | 1 | 7 | values hidden |
| 4 | Unnamed: 4 | 0 | 4 | values hidden |
| 5 | TEXT 1^READING | 0 | 16 |  |
| 6 | Unnamed: 6 | 0 | 16 | values hidden |
| 7 | TEXT 2^READING | 0 | 17 |  |
| 8 | Unnamed: 8 | 0 | 17 | values hidden |
| 9 | TEXT 3^READING | 0 | 16 |  |
| 10 | Unnamed: 10 | 0 | 16 | values hidden |

**Sheet `Foglio2`** -- 0 rows x 0 columns (fully-empty rows: 0, redacted headers: 0)

| # | Column | Missing | Unique | Summary (only non-sensitive) |
|---|---|---|---|---|

**Sheet `Foglio3`** -- 0 rows x 0 columns (fully-empty rows: 0, redacted headers: 0)

| # | Column | Missing | Unique | Summary (only non-sensitive) |
|---|---|---|---|---|

### `excel_file_2:Tab 3.xlsx` (readable: True, error: None)

**Sheet `Sheet1`** -- 29 rows x 11 columns (fully-empty rows: 3, redacted headers: 0)

| # | Column | Missing | Unique | Summary (only non-sensitive) |
|---|---|---|---|---|
| 0 | PERSON | 3 | 24 |  |
| 1 | Unnamed: 1 | 6 | 15 | values hidden |
| 2 | Unnamed: 2 | 6 | 3 | values hidden |
| 3 | Unnamed: 3 | 6 | 13 | values hidden |
| 4 | Unnamed: 4 | 6 | 2 | values hidden |
| 5 | TEXT 1^READING | 6 | 23 |  |
| 6 | Unnamed: 6 | 6 | 23 | values hidden |
| 7 | TEXT 2^READING | 6 | 23 |  |
| 8 | Unnamed: 8 | 6 | 23 | values hidden |
| 9 | TEXT 3^READING | 6 | 23 |  |
| 10 | Unnamed: 10 | 6 | 23 | values hidden |

### `excel_file_3:TAB 5.xlsx` (readable: True, error: None)

**Sheet `Foglio1`** -- 30 rows x 11 columns (fully-empty rows: 0, redacted headers: 0)

| # | Column | Missing | Unique | Summary (only non-sensitive) |
|---|---|---|---|---|
| 0 | Unnamed: 0 | 26 | 4 | values hidden |
| 1 | Unnamed: 1 | 1 | 20 | values hidden |
| 2 | Unnamed: 2 | 1 | 10 | values hidden |
| 3 | Unnamed: 3 | 1 | 3 | values hidden |
| 4 | Unnamed: 4 | 1 | 18 | values hidden |
| 5 | TEXT 1^ READING | 0 | 30 |  |
| 6 | Unnamed: 6 | 0 | 30 | values hidden |
| 7 | TEXT 2^ READING | 0 | 26 |  |
| 8 | Unnamed: 8 | 0 | 26 | values hidden |
| 9 | WORDS 1^ READING | 0 | 30 |  |
| 10 | Unnamed: 10 | 0 | 30 | values hidden |

**Sheet `Foglio2`** -- 0 rows x 0 columns (fully-empty rows: 0, redacted headers: 0)

| # | Column | Missing | Unique | Summary (only non-sensitive) |
|---|---|---|---|---|

**Sheet `Foglio3`** -- 0 rows x 0 columns (fully-empty rows: 0, redacted headers: 0)

| # | Column | Missing | Unique | Summary (only non-sensitive) |
|---|---|---|---|---|

### Linkage to participants

**Join status: PARTIAL**  participants by status: {'not_linked': 60, 'linked_unambiguous': 1}

| File | Sheet | Linked | Key column | Key type | Matched (any) | Exactly 1 row | Multiple rows | Age col(s) | Sex col(s) |
|---|---|---|---|---|---|---|---|---|---|
| excel_file_1:15 YHC.xlsx | Foglio1 | True | Unnamed: 0 | full | 1 | 1 | 0 | [] | [] |
| excel_file_2:Tab 3.xlsx | Sheet1 | False | n/a | n/a | 0 | n/a | n/a | n/a | n/a |
| excel_file_3:TAB 5.xlsx | Foglio1 | False | n/a | n/a | 0 | n/a | n/a | n/a | n/a |

## 11. Manifest summary

- Rows: 831 | columns: ['subject_id', 'recording_id', 'label', 'is_pd', 'age_if_available', 'sex_if_available', 'metadata_link_status', 'task_code', 'task_name', 'task_family', 'task_parse_status', 'audio_path', 'duration_sec', 'sample_rate', 'channels', 'num_frames', 'subtype', 'readable', 'peak_abs', 'rms_dbfs', 'clip_fraction', 'silence_fraction', 'qc_flags', 'file_sha256', 'audio_sha256']
- Rows with age: 0 | rows with sex: 0
- audio_path column contains: **pseudonymous logical paths (real paths only in private/path_lookup_PRIVATE.csv)**

## 12. What YOU must verify by hand

- [ ] Open 3 or 4 participant folders at random and confirm each folder is ONE person (not a family folder or a mix of people).
- [ ] Confirm no person appears in two folders under different spellings (the script can only detect identical names / identical audio).
- [ ] Check the label-vs-metadata cross-tab in section 10 (if shown): do folder labels agree with the Excel group column?
- [ ] Confirm the Excel columns used for age/sex (section 10) really are age and sex, and that the sex coding is what you expect.
- [ ] Check the filename shapes (section 6) look like what you see in the file explorer; if many files failed task decoding, send me 5 name-REDACTED example filenames.
- [ ] Listen to 2 or 3 WAVs flagged by QC (if any) and to one WAV per task type to confirm the task code matches what is spoken.
- [ ] Confirm the sample-rate / channel summary matches the properties you saw earlier in your file explorer.
- [ ] Keep the private folder out of Git and out of any chat or upload.
