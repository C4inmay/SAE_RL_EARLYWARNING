# Notebook 03 — ECG and HRV Feature Extraction Setup

## Purpose

This notebook establishes the starting point for linking the sepsis ICU cohort to MIMIC waveform records and extracting ECG-derived heart-rate-variability (HRV) features. Its stated intended pipeline is:

`ECG waveform → preprocessing → R-peak detection → RR intervals → HRV features → hourly aggregation`

## What was completed

1. **Configured project paths.** The notebook defined `data/processed` as the input/output location for processed cohort data and `data/raw/mimic-waveform` as the intended waveform-data location.
2. **Created the waveform directory if absent.** This prepares a local folder but does not download, copy, or verify waveform recordings.
3. **Loaded the prepared sepsis cohort.** It read `sepsis_icu_cohort_demo.csv` and extracted the unique candidate subjects for waveform matching.

## Results

- The cohort contains **17 unique patients** eligible for waveform-record linkage.
- The notebook recorded the 17 subject IDs as the waveform-matching candidate set.
- No waveform files, ECG channels, sampling rates, R peaks, RR intervals, HRV values, hourly HRV table, or merged RL dataset were produced in this notebook.

## Current dependencies

- Input cohort: `data/processed/sepsis_icu_cohort_demo.csv` from Notebook 01.
- Intended waveform location: `data/raw/mimic-waveform/`.
- Candidate list already available: `data/processed/sepsis_waveform_candidates.csv`.

## Work still required to complete the stated objective

1. Obtain or mount the appropriate MIMIC waveform database and its record metadata.
2. Match cohort admissions/stays to waveform records using the available linkage metadata and timing, rather than subject ID alone.
3. Select a valid ECG lead/channel and load each record with its sampling frequency and start time.
4. Apply ECG signal-quality checks and preprocessing (for example, filtering and artifact handling).
5. Detect R peaks, reject implausible beats/RR intervals, and calculate validated HRV measures such as mean NN, SDNN, RMSSD, pNN50, and frequency-domain features where recording length supports them.
6. Aggregate valid features into the same ICU-hour time base used in `clinical_hourly_demo.csv`.
7. Join HRV features to the clinical hourly table, retain feature-availability indicators, and document waveform coverage and exclusions.

## Interpretation

This is a setup/scaffolding notebook rather than a completed ECG/HRV extraction pipeline. The `mimic-waveform` directory being present only confirms a local target path; it is not evidence that waveform data or derived HRV features are available.
