# Implementation Plan: Night Data Preparation

## Overview

Python 3.10 (conda env `mestrado`) and bash. Tests in `code/tests/` with `pytest` and `hypothesis`, always on temporary copies, never on the real `data/` or `util/` folders. STR is out of scope. Stage 2 code is not changed.

## Tasks

- [x] 1. Test infrastructure
  - Create `code/tests/` with `conftest.py` providing a fixture that builds a small synthetic dataset in `tmp_path`: a labels CSV with 3 nights (one EPB, one quiet, one cloudy with `has_epb` empty), matching 512×512 grayscale PNGs, and a `util/NIGHT/processed/` tree with some unlabeled frames.
  - Ensure `pytest` and `hypothesis` are installed in `mestrado`; add a `code/pytest.ini` pointing to `tests/`.
  - _Requirements: 1, 4, 5_

- [x] 2. Locked test split
  - [x] 2.1 Implement `load_test_nights` and the `split` parameter of `load_raw_dataset`
    - In `epb_detection_system_v5.py`: `load_test_nights(path)` parses comments and blank lines and returns an empty set if the file is missing.
    - Add `split="dev" | "test" | "all"` and `test_nights_path` to `load_raw_dataset`; compute nights with `extract_night_groups`; log excluded frames/nights; warn for listed nights absent from the CSV; reject unknown `split` values.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_
  - [x] 2.2 Write property test for the split partition
    - **Property 1: Dev and test splits partition the data by night**
    - **Validates: Requirements 1.3, 1.4, 1.5**
  - [x] 2.3 Implement `evaluate_locked_test`
    - Train on `split="dev"`, predict each test night, return per-night accuracy and pooled `_binary_metrics`; append a JSON line to `data/test_evaluations.log` (timestamp, `git rev-parse HEAD`, nights, note, metrics); raise on empty test set.
    - _Requirements: 1.7, 1.8, 1.9_
  - [x] 2.4 Write unit tests for `load_test_nights` and `evaluate_locked_test`
    - Comments/blank lines, missing file, warning for unknown night, empty test set raises, one log line appended per call.
    - _Requirements: 1.1, 1.2, 1.6, 1.8, 1.9_
  - [x] 2.5 Create `data/test_nights.txt`
    - Instructions in Portuguese (format `SITE_YYYYMMDD`, choose before seeing results, never edit after first locked evaluation) and an empty list.
    - _Requirements: 1.10_

- [x] 3. Checkpoint - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Clean images for existing nights
  - [x] 4.1 Add `--offline` and `--clean-only` to `util/embrace.sh`
    - Parse flags before the URL/night argument; `--offline` skips listing and download and uses TIFs in `original/`; `--clean-only` writes only `clean/`, skips `processed/` and the timelapse. Without flags, behavior is unchanged.
    - _Requirements: 2.1, 2.2, 2.3_
  - [x] 4.2 Create `util/regen_clean.sh`
    - Loop over `util/*/original/`, call `embrace.sh --offline --clean-only`, continue on failure, print per-night TIF/clean/processed counts, exit non-zero if any night has fewer `clean/` files than TIFs.
    - _Requirements: 2.4, 2.5, 2.6_
  - [x] 4.3 Create `util/check_clean.py`
    - For every frame, compare `clean/` with `processed/` inside the circular field of view (radius 0.44); report frames differing by more than 1 gray level; exit non-zero if any.
    - _Requirements: 2.7_
  - [x] 4.4 Write offline test for `embrace.sh` flags
    - On 2 copied TIFs in `tmp_path`: `--offline --clean-only` creates `clean/` and leaves `processed/` empty and no video; `check_clean.py` passes against a full run.
    - _Requirements: 2.1, 2.2, 2.7_
  - [x] 4.5 Run `regen_clean.sh` and `check_clean.py` on the real `util/` folders
    - Report per-night counts and any failing frame to the user.
    - _Requirements: 2.4, 2.6, 2.7_

- [x] 5. Labeling protocol and agreement check
  - [x] 5.1 Write `data/LABELING_PROTOCOL.md`
    - One page in Portuguese: Verdicts mapped to CSV columns, partial-cloud rule, event start/end rule, 15–20 frames per night plus the 30 minutes around events, test-night rule, how to run the agreement check and how to read kappa.
    - _Requirements: 3.1_
  - [x] 5.2 Implement `util/label_agreement.py`
    - `sample --n --seed`: proportional per night, at least 2 per night, shuffled, blank labels.
    - `score FILE`: Verdict from (`has_epb`, `cloudy`), confusion table, raw agreement, Cohen's kappa overall and per night, disagreements CSV, count of ignored blank rows.
    - _Requirements: 3.2, 3.3, 3.4, 3.5_
  - [x] 5.3 Write property test for kappa
    - **Property 5: Agreement is symmetric and maximal for identical labels**
    - **Validates: Requirements 3.3**
  - [x] 5.4 Write unit tests for `label_agreement.py`
    - Sample has blank labels and ≥2 frames per night; kappa matches a hand-computed table; blank rows are ignored.
    - _Requirements: 3.2, 3.3, 3.5_

- [x] 6. Checkpoint - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Labeling queue and merge
  - [x] 7.1 Implement `util/label_queue.py`
    - Scan `util/*/processed/*.png`, exclude labeled frames and `--exclude-sites` (default `STR`), evenly spaced selection with `--per-night` (default 20) including first and last, `--all`, refuse to overwrite a filled queue without `--force`, write `data/label_queue.csv`, contact sheets in `data/label_queue_sheets/`, summary table.
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_
  - [x] 7.2 Write property test for the queue
    - **Property 4: The queue never contains labeled frames**
    - **Validates: Requirements 4.1, 4.2**
  - [x] 7.3 Implement `util/merge_labels.py`
    - Skip unlabeled rows, validate all rows first (all-or-nothing), timestamped backup, append rows with `site`, `datetime`, `filter` from the filename, copy PNGs without overwriting, print the metadata commands.
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_
  - [x] 7.4 Write property test for merge preserving existing rows
    - **Property 2: Merging labels never alters existing rows**
    - **Validates: Requirements 5.5**
  - [x] 7.5 Write property test for invalid input
    - **Property 3: Invalid input leaves the CSV untouched**
    - **Validates: Requirements 5.2, 5.3**
  - [x] 7.6 Make `util/moon_illumination.py` fail on unknown stations
    - `get_site_coordinates` raises `ValueError` naming the station instead of returning 0,0.
    - _Requirements: 5.8_
  - [x] 7.7 Write unit tests for queue, merge and moon script
    - Even spacing picks first/last; STR excluded by default; `--force` behavior; duplicate filename rejected; existing image not overwritten; unknown station raises.
    - _Requirements: 4.3, 4.8, 5.2, 5.6, 5.8_
  - [x] 7.8 Generate the real labeling queue
    - Run `label_queue.py` on the real `util/` and show the user the per-night summary.
    - _Requirements: 4.7_

- [-] 8. Final checkpoint
  - Re-run `epb_detection_system_v5.ipynb` with the empty `test_nights.txt` and confirm the results are unchanged.
  - Ensure all tests pass, then commit, and ask the user if questions arise.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1", "2.5", "4.1", "5.1", "7.6"] },
    { "id": 1, "tasks": ["2.1", "4.2", "4.3", "5.2", "7.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "4.4", "5.3", "5.4", "7.2", "7.3"] },
    { "id": 3, "tasks": ["2.4", "4.5", "7.4", "7.5", "7.7", "7.8"] }
  ]
}
```

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster first pass.
- Each task references the requirements it covers; property tests reference the design properties.
- Nothing in this plan writes to `data/epb_labels.csv` except `merge_labels.py`, and only when the user runs it on a filled queue.
