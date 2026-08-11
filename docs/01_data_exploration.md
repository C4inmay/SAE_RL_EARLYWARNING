# Notebook 01 — Data Exploration and Sepsis-Cohort Preparation

## Purpose

This notebook explored the locally available MIMIC-IV demo data and prepared the first project datasets for an RL-based early-warning study of neurological deterioration in sepsis ICU patients. It established the available data tables, identified clinical measurement codes, defined a diagnosis-based sepsis cohort, examined GCS trajectories, and saved reusable processed files.

## Inputs

- `data/raw/mimic-iv/hosp/patients.csv.gz`
- `data/raw/mimic-iv/hosp/admissions.csv.gz`
- `data/raw/mimic-iv/hosp/diagnoses_icd.csv.gz`
- `data/raw/mimic-iv/hosp/d_icd_diagnoses.csv.gz`
- `data/raw/mimic-iv/icu/icustays.csv.gz`
- `data/raw/mimic-iv/icu/chartevents.csv.gz`
- `data/raw/mimic-iv/icu/d_items.csv.gz`

## What was done

1. **Inspected the MIMIC-IV directory structure.** The hospital (`hosp`) and ICU (`icu`) modules were enumerated to confirm which compressed CSV tables were present.
2. **Loaded and profiled core tables.** `patients`, `admissions`, and `icustays` were loaded and their schemas, example rows, and sizes were inspected. The local demo extract contained 100 patients, 275 admissions, and 140 ICU stays.
3. **Inspected clinical chart events.** A 10,000-row sample of `chartevents` was loaded to inspect its event-level structure: identifiers, chart and store times, item IDs, textual/numeric values, units, and warnings. The sample contained 554 distinct item IDs.
4. **Mapped measurement item IDs.** The ICU item dictionary (`d_items`) was searched for vital-sign, neurological, and ECG-related concepts. This identified the measurements used downstream:

   | Variable | MIMIC item ID | Unit / meaning |
   | --- | ---: | --- |
   | Heart rate | 220045 | bpm |
   | Mean arterial pressure (MAP) | 220052 | mmHg; arterial BP mean |
   | Respiratory rate | 220210 | inspirations/minute |
   | GCS motor | 223901 | neurological component |
   | GCS verbal | 223900 | neurological component |
   | GCS eye opening | 220739 | neurological component |

   The dictionary had 4,014 item definitions. Searching for “glasgow” alone returned no rows, while searching for “GCS” found the component measurements above. The ECG-related dictionary entries found were Heart Rhythm (220048) and EKG (225402); they are charted concepts, not raw waveform samples.
5. **Built a diagnosis-based sepsis cohort.** Diagnosis records were joined to the ICD dictionary on `icd_code` and `icd_version`. Rows whose diagnosis title contained `sepsis` or `septic` were selected, then their subjects were linked to `icustays`.
6. **Characterized ICU length of stay.** The cohort LOS distribution was summarized and plotted.
7. **Extracted target clinical events.** The full local `chartevents` file (668,862 rows) was loaded with only required columns. Events were restricted to the cohort’s 33 ICU stays and the six selected vital/GCS item IDs, then mapped to readable feature names and sorted chronologically.
8. **Constructed GCS trajectories.** The three GCS components were pivoted into one timestamp-level table. Total GCS was calculated only where all three components were available: `eye + verbal + motor`. The notebook then assessed missingness, score distribution, per-stay coverage, sequential score changes, and candidate deterioration thresholds.

## Results

### Sepsis cohort

- 52 sepsis-related diagnosis rows were identified.
- These represented 17 unique patients and 24 admissions.
- Linking those patients to ICU stays yielded **33 ICU stays** from **17 patients**.
- ICU LOS (days): mean **5.88**, median **3.61**, minimum **0.64**, maximum **20.53**.

### Clinical-event coverage

- **16,718** relevant clinical events were extracted across all 33 stays and 17 patients.
- Counts by feature: heart rate 5,414; respiratory rate 5,414; MAP 2,582; GCS eye 1,106; GCS verbal 1,103; GCS motor 1,099.
- Heart rate, respiratory rate, and all three GCS components covered every cohort stay; MAP covered 15 stays from 12 patients.

### GCS trajectory results

- The pivoted GCS table contains **1,107 timestamped rows** across all cohort stays.
- 1,098 complete GCS totals were available; 9 rows were incomplete.
- Total GCS: mean **10.96**, median **11**, range **3–15**.
- Of 1,060 sequential non-missing GCS changes, the mean change was **+0.09** and the largest observed fall was **−9** points.
- Using a candidate deterioration definition of `ΔGCS <= −3`, the raw timestamp-level analysis found **33 events**, spanning **12 ICU stays** and **10 patients**.

## Files produced

- `data/processed/sepsis_icu_cohort_demo.csv` — 33-row ICU cohort with identifiers, care units, ICU entry/exit times, and LOS.
- `data/processed/sepsis_waveform_candidates.csv` — 17 unique cohort subject IDs for later waveform matching.
- `data/processed/sepsis_clinical_events_demo.csv` — 16,718 filtered long-format clinical events with feature labels.
- `data/processed/sepsis_gcs_trajectory_demo.csv` — 1,107 timestamp-level GCS component/total records.

## Important interpretation notes

- Sepsis was defined by a text search over ICD diagnosis titles. It is suitable for this demo/research pipeline but is not a formal clinical phenotype such as Sepsis-3.
- GCS is charted intermittently, so a change is a difference between consecutive recorded scores, not necessarily between fixed time intervals.
- The selected MAP item is arterial MAP only; it does not include non-invasive MAP measurements.
- This notebook does not use raw ECG waveforms; it only prepares candidate subject IDs and identifies relevant charted ECG concepts.
