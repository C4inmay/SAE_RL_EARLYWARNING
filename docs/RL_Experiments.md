# SAE Early Warning — Reinforcement Learning Experiments

## 1. Overview

This document summarizes the Reinforcement Learning experiments performed for the
SAE (Serious Adverse Event) early-warning component of the project.

The objective was to investigate whether an RL agent could learn to issue an
early warning when an SAE was expected in the following hour.

The experiments used the processed MIMIC-III clinical dataset developed during
the earlier stages of the project.

---

# 2. RL Dataset

The final RL-ready dataset contained:

- Total hourly observations: **4,556**
- Patients: **25**
- ICU stays: **38**
- SAE observations: **35**
- SAE prevalence: **0.77%**

The target used for RL was:

> `future_sae_1h` — whether an SAE occurs in the following hour.

The dataset contained severe class imbalance:

| Class | Observations |
|---|---:|
| No future SAE | 4,521 |
| Future SAE | 35 |

---

# 3. Patient-Level Train/Test Split

To prevent patient-level data leakage, the dataset was divided by patient rather
than randomly splitting individual hourly observations.

### Training Set

- Patients: **20**
- ICU stays: **33**
- Hourly rows: **3,302**
- Future SAE events: **22**

### Test Set

- Patients: **5**
- ICU stays: **5**
- Hourly rows: **1,254**
- Future SAE events: **13**

### Leakage Check

Patient overlap between training and testing:

**0 patients**

Therefore, the final test patients were unseen during training.

---

# 4. RL State Representation

The RL environment used seven clinical/time features:

1. `gcs_last_observed`
2. `previous_observed_gcs`
3. `heart_rate`
4. `map`
5. `resp_rate`
6. `spo2`
7. `hour`

Missing values were filled using medians calculated from the training data.

The resulting observation space was:

```text
Box(-inf, inf, (7,), float32)


5. RL Action Space

Three warning actions were defined:

Action	Meaning
0	No warning / routine monitoring
1	Moderate warning
2	High-risk warning / escalation

The actions represent a constructed warning policy rather than historical
clinical interventions.

6. Initial DQN Experiment

The first experiment used a standard Deep Q-Network (DQN).

The model was trained on the real MIMIC-III training cohort and evaluated on
the completely unseen real MIMIC-III test cohort.

Results
Model	Precision	Recall	F1
Original DQN	0.0588	0.3333	0.1000

The confusion matrix was:

[[119, 16],
 [  2,  1]]

The DQN detected 1 of the 3 SAE events in the evaluated episode.

This established the initial RL baseline.

7. Reward-Shaped DQN Experiment

The second experiment modified the reward structure to place a larger penalty
on missing an SAE event.

The objective was to determine whether the original DQN's limited recall was
primarily caused by an insufficiently strong reward signal.

Results
Model	Precision	Recall	F1
Reward-Shaped DQN	0.0000	0.0000	0.0000

Confusion matrix:

[[1241, 0],
 [  13, 0]]

The model predicted no positive SAE warnings.

Although the accumulated reward was high:

Total reward = 1176.0

this did not translate into improved SAE detection.

This demonstrates that maximizing the current reward does not necessarily
produce a clinically useful warning policy under severe class imbalance.

8. Synthetic Trajectory Augmentation

A synthetic dataset was generated to provide additional temporal deterioration
trajectories.

The synthetic data was designed as patient trajectories rather than independent
random rows.

The generated physiological variables included:

GCS
Previous GCS
Heart rate
MAP
Respiratory rate
SpO2
Hour
Future SAE target
Synthetic Dataset
Synthetic patients: 30
Synthetic hourly rows: 3,600
Synthetic future SAE events: 11

The synthetic data was used only for training.

The real MIMIC-III test cohort remained completely untouched.

Augmented Training Dataset
Dataset	Rows	Future SAE
Real training data	3,302	22
Synthetic data	3,600	11
Combined training data	6,902	33
9. Synthetic-Augmented DQN

The synthetic trajectories were combined with the real training data and used
to train another DQN.

The model was still evaluated exclusively on the real unseen MIMIC-III test
patients.

Results
Model	Precision	Recall	F1
Synthetic-Augmented DQN	0.0000	0.0000	0.0000

Confusion matrix:

[[1240, 1],
 [  13, 0]]

The synthetic augmentation therefore did not improve SAE detection.

This experiment suggests that simply increasing the number of training
trajectories with synthetic data is insufficient to solve the learning problem.

10. Cost-Sensitive DQN Experiment

A third strategy was introduced to explicitly account for the high cost of
missing an SAE event.

The reward structure was:

Situation	Reward
Correctly predicting no SAE	+0.1
Correctly detecting SAE	+10
False alarm	-1
Missed SAE	-10

The purpose was to make SAE misses substantially more costly than normal
observations.

Results
Model	Precision	Recall	F1
Cost-Sensitive DQN	0.0000	0.0000	0.0000

Confusion matrix:

[[1217, 24],
 [  13,  0]]

The model produced 24 warning actions but none corresponded to the 13 actual
future SAE events.

Action distribution:

Action	Count	Percentage
0	1,230	98.09%
2	24	1.91%

The total test reward was:

-32.3

Therefore, cost-sensitive reward design also failed to improve SAE detection.

11. Overall Comparison

The experiments produced the following results:

Model	Precision	Recall	F1	Total Reward
Original DQN	0.0588	0.3333	0.1000	82.0
Reward-Shaped DQN	0.0000	0.0000	0.0000	1176.0
Synthetic-Augmented DQN	0.0000	0.0000	0.0000	1174.5
Cost-Sensitive DQN	0.0000	0.0000	0.0000	-32.3
12. Interpretation

The initial DQN achieved the best F1-score among the tested RL approaches:

F1 = 0.10

and achieved:

Recall = 33.3%

However, the subsequent reward-shaping, synthetic augmentation, and
cost-sensitive experiments did not improve the result.

The experiments therefore indicate that the main difficulty is not simply the
choice of reward function or the number of training rows.

The problem is influenced by several factors:

The dataset contains only 25 patients.
There are only 35 SAE events.
SAE prevalence is approximately 0.77%.
The RL state contains only seven clinical/time variables.
The warning actions are constructed actions rather than historical
clinician interventions.
The available MIMIC-III cohort used in this project is a very small
demonstration cohort.
13. Important Limitation

The current results should be considered a proof-of-concept RL experiment
rather than evidence of clinical effectiveness.

The test cohort contains only:

5 patients
5 ICU stays
1,254 hourly observations
13 future SAE events

Consequently, a small change in the number of detected events can substantially
change precision, recall and F1.

The results should therefore not be interpreted as clinically validated
performance.

14. Synthetic Data Limitation

Synthetic data was used only as a training augmentation experiment.

The final test set remained completely real and consisted of unseen MIMIC-III
patients.

The synthetic experiment did not improve the test performance, and therefore
synthetic data should not be presented as evidence of improved clinical
performance.

15. Final RL Conclusion

The RL pipeline was successfully implemented from clinical preprocessing through
patient-level splitting, environment construction, DQN training and evaluation.

Four approaches were evaluated:

Original DQN
Reward-shaped DQN
Synthetic-augmented DQN
Cost-sensitive DQN

The Original DQN produced the strongest result, with an F1-score of
0.10 and recall of 33.3%.

The subsequent experiments did not improve upon this result.

The primary limitation is the very small number of patients and SAE events
available in the MIMIC-III demonstration cohort. A substantially larger,
credentialed clinical dataset would be required for stronger validation.

16. Files Produced

The RL experiments produced the following important artifacts:

notebooks/
├── sae_rl_experiment.ipynb
├── sae_rl_improvement_experiments.ipynb
└── rl_cost_sensitive_experiment.ipynb


models/
├── dqn_sae_warning
├── dqn_sae_reward_shaped
├── dqn_sae_synthetic_augmented
└── dqn_cost_sensitive_sae


data/processed/mimic3/
├── sae_rl_train.csv
├── sae_rl_test.csv
├── sae_rl_results.csv
├── sae_rl_improvement_results.csv
└── sae_rl_cost_sensitive_results.csv


data/processed/mimic3/synthetic/
└── synthetic_sae_trajectories.csv