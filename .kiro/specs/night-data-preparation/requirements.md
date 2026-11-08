# Requirements Document

## Introduction

Prepare the EPB dataset and tooling for adding new observation nights from EMBRACE, before further changes to the Stage 2 classifier. The scope is: a locked test split by night, clean images (no burned-in text) for the nights already on disk, a labeling protocol with an agreement check, and a labeling queue built from unlabeled frames already in `code/util/`. The STR station is out of scope. Stage 2 code and training data are not changed.

## Glossary

- **Night**: an observation night, identified as `SITE_YYYYMMDD` with the start date; frames before 12:00 UT belong to the previous date (as in `extract_night_groups`).
- **Test_Split**: the set of nights listed in `data/test_nights.txt`, reserved for the final evaluation.
- **Dev_Split**: all labeled frames whose night is not in the Test_Split.
- **Labels_CSV**: `code/data/epb_labels.csv`.
- **Loader**: `load_raw_dataset` in `epb_detection_system_v5.py`.
- **Locked_Evaluator**: `evaluate_locked_test` in `epb_detection_system_v5.py`.
- **Embrace_Script**: `code/util/embrace.sh`.
- **Regen_Script**: `code/util/regen_clean.sh`.
- **Queue_Tool**: `code/util/label_queue.py`.
- **Merge_Tool**: `code/util/merge_labels.py`.
- **Agreement_Tool**: `code/util/label_agreement.py`.
- **Moon_Script**: `code/util/moon_illumination.py`.
- **Verdict**: the three-class label `EPB`, `Sem EPB` or `Indeterminado` (cloudy).

## Requirements

### Requirement 1: Locked test split by night

**User Story:** As a researcher, I want some nights reserved for a single final evaluation, so that the reported result is not biased by design decisions made on the same nights.

#### Acceptance Criteria

1. THE system SHALL read the Test_Split from `data/test_nights.txt`, one night id per line, ignoring blank lines and lines starting with `#`.
2. IF `data/test_nights.txt` does not exist, THEN THE Loader SHALL treat the Test_Split as empty.
3. WHEN the Loader is called with `split="dev"` (the default), THE Loader SHALL return only frames whose night is not in the Test_Split.
4. WHEN the Loader is called with `split="test"`, THE Loader SHALL return only frames whose night is in the Test_Split.
5. WHEN the Loader is called with `split="all"`, THE Loader SHALL return all labeled frames regardless of the Test_Split.
6. WHEN a night listed in the Test_Split has no frames in the Labels_CSV, THE Loader SHALL log a warning naming that night.
7. WHEN the Locked_Evaluator runs, THE Locked_Evaluator SHALL train on the Dev_Split, predict each Test_Split night, and return per-night accuracy and pooled metrics.
8. WHEN the Locked_Evaluator runs, THE Locked_Evaluator SHALL append one line to `data/test_evaluations.log` with the timestamp, git commit hash, Test_Split nights, a user note and the metrics.
9. IF the Test_Split is empty, THEN THE Locked_Evaluator SHALL raise an error without evaluating.
10. THE system SHALL create `data/test_nights.txt` with usage instructions and an empty night list.

### Requirement 2: Clean images for existing nights

**User Story:** As a researcher, I want text-free images for every night already downloaded, so that future models do not see date and time stamps.

#### Acceptance Criteria

1. WHEN the Embrace_Script is called with `--offline`, THE Embrace_Script SHALL skip the listing and download steps and process the TIFs already in `original/`.
2. WHEN the Embrace_Script is called with `--clean-only`, THE Embrace_Script SHALL write only `clean/` PNGs and SHALL NOT modify `processed/` or create a timelapse.
3. THE Embrace_Script SHALL keep its current behavior when called without the new flags.
4. WHEN the Regen_Script runs, THE Regen_Script SHALL call the Embrace_Script with `--offline --clean-only` for every `util/*/original/` folder.
5. WHEN a night fails, THE Regen_Script SHALL continue with the remaining nights and report all failures at the end.
6. WHEN the Regen_Script finishes, THE Regen_Script SHALL print, per night, the counts of TIF, `clean/` and `processed/` files, and exit with an error if any night has fewer `clean/` files than TIFs.
7. THE system SHALL verify that, inside the circular field of view, each `clean/` image differs from its `processed/` counterpart by at most 1 gray level, and list every frame that fails.

### Requirement 3: Labeling protocol and agreement check

**User Story:** As a researcher, I want written label definitions and a measure of my own consistency, so that label noise is known and reported.

#### Acceptance Criteria

1. THE system SHALL provide `data/LABELING_PROTOCOL.md`, in Portuguese, defining the Verdicts, the partial-cloud rule, the event start/end rule, the per-night sampling rule and the rule for choosing test nights before seeing model results.
2. WHEN the Agreement_Tool runs `sample --n N --seed S`, THE Agreement_Tool SHALL write `data/relabel_sample.csv` with N labeled frames drawn proportionally per night, at least 2 per night, in shuffled order, with the label columns empty.
3. WHEN the Agreement_Tool runs `score FILE`, THE Agreement_Tool SHALL compare the re-labels with the Labels_CSV on the Verdict and print the confusion table, raw agreement and Cohen's kappa overall and per night.
4. WHEN the Agreement_Tool runs `score FILE`, THE Agreement_Tool SHALL write the disagreeing frames to `data/relabel_disagreements.csv`.
5. THE Agreement_Tool SHALL ignore rows of FILE whose labels are still empty, and report how many were ignored.

### Requirement 4: Labeling queue from existing nights

**User Story:** As a researcher, I want a list of unlabeled frames already on disk, spread over each night, so that I can label many nights quickly.

#### Acceptance Criteria

1. WHEN the Queue_Tool runs, THE Queue_Tool SHALL scan `util/*/processed/*.png` and exclude every frame already in the Labels_CSV.
2. THE Queue_Tool SHALL exclude stations listed in `--exclude-sites`, with `STR` excluded by default.
3. WHEN `--per-night K` is given (default 20), THE Queue_Tool SHALL select, per night, up to K unlabeled frames evenly spaced in time, including the first and last unlabeled frame of the night.
4. WHEN `--all` is given, THE Queue_Tool SHALL queue every unlabeled frame.
5. THE Queue_Tool SHALL write `data/label_queue.csv` with columns `filename, night, site, path, has_epb, cloudy, notes`, with the label columns empty.
6. THE Queue_Tool SHALL write one contact sheet per night to `data/label_queue_sheets/NIGHT.png`, with the queued frames in time order and their time stamps.
7. THE Queue_Tool SHALL print, per night, the number of frames on disk, already labeled and queued.
8. IF `data/label_queue.csv` exists and contains any filled label, THEN THE Queue_Tool SHALL refuse to overwrite it unless `--force` is given.

### Requirement 5: Merging labels into the dataset

**User Story:** As a researcher, I want to add my new labels to the dataset safely, so that I never lose or corrupt existing labels.

#### Acceptance Criteria

1. WHEN the Merge_Tool runs on a filled queue, THE Merge_Tool SHALL skip rows where both `has_epb` and `cloudy` are empty.
2. THE Merge_Tool SHALL reject a row IF `has_epb` is not in {0, 1, empty}, OR `cloudy` is not in {0, 1, empty}, OR `cloudy=1` with `has_epb` filled, OR `cloudy=0` with `has_epb` empty, OR the filename is already in the Labels_CSV.
3. IF any row is rejected, THEN THE Merge_Tool SHALL list every rejected row and exit without modifying any file.
4. WHEN all rows are valid, THE Merge_Tool SHALL write a timestamped backup of the Labels_CSV before appending.
5. WHEN all rows are valid, THE Merge_Tool SHALL append the rows with `filename, has_epb, cloudy, site, datetime, filter`, leaving the other columns empty, and SHALL NOT change existing rows.
6. WHEN all rows are valid, THE Merge_Tool SHALL copy each PNG into `data/all-images/` without overwriting an existing file.
7. WHEN the merge finishes, THE Merge_Tool SHALL print the commands to fill the remaining metadata.
8. IF the Moon_Script finds a station without coordinates, THEN THE Moon_Script SHALL raise an error naming the station instead of using latitude/longitude 0,0.
