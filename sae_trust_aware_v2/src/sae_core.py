"""
SAE Trust-Aware Alarm Controller v2

Research pipeline:
  Stage 1: supervised calibrated risk forecaster
  Stage 2: offline Fitted-Q Iteration (FQI) alarm controller
  Game: clinician trust is a state variable for trust-aware policies.

Important design choices in v2:
  - train/validation/test separation for risk calibration
  - explicit ablation:
      fixed_no_trust_state
      fixed_with_trust_state
      trust_aware
  - threshold selection is performed on validation, never on test
  - effective detection is reported separately from raised detection
  - synthetic cohort is only a mechanism/proof-of-concept
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score


WAIT, ALARM = 0, 1


@dataclass
class Config:
    horizon: int = 6
    n_features: int = 5
    event_rate: float = 0.35
    min_T: int = 24
    max_T: int = 60

    # Trust dynamics
    trust_dec_fa: float = 0.28
    trust_rec: float = 0.015
    trust_rec_true: float = 0.04
    act_floor: float = 0.05

    # Rewards
    r_hit: float = 3.0
    r_fa: float = 1.0
    r_miss: float = 2.0
    r_tn: float = 0.05
    wait_cost: float = 0.0
    gamma: float = 0.98
    kappa: float = 0.5
    ignored_true_fraction: float = 0.15

    # FQI
    fqi_iters: int = 15
    n_trees: int = 80
    min_samples_leaf: int = 8

    # Reproducibility
    seed: int = 0
    feature_weights: tuple = (1.0, 0.8, 0.6, 1.2, 0.9)


def simulate_cohort(
    n_patients: int,
    cfg: Config,
    rng: np.random.Generator,
    noise: float = 1.05,
    ramp_mag: float = 2.0,
    confound: float = 0.0,
) -> List[dict]:
    """Synthetic ICU trajectories used only for mechanism validation."""
    patients = []
    for pid in range(n_patients):
        T = int(rng.integers(cfg.min_T, cfg.max_T + 1))
        is_event = rng.random() < cfg.event_rate
        tau = int(rng.integers(cfg.horizon + 2, T)) if (
            is_event and T > cfg.horizon + 3
        ) else None

        sev = rng.normal(0.0, 0.25, size=T).cumsum() * 0.12
        sev = sev - sev.mean()

        if tau is None and confound > 0:
            sev = sev + confound

        if tau is not None:
            ramp_start = max(0, tau - 7)
            ramp = np.clip((np.arange(T) - ramp_start) / 6.0, 0, 1.0) * ramp_mag
            sev = sev + ramp

        W = np.asarray(cfg.feature_weights[:cfg.n_features], dtype=np.float32)
        X = sev[:, None] * W[None, :] + rng.normal(0, noise, size=(T, cfg.n_features))

        patients.append(
            {
                "patient_id": pid,
                "X": X.astype(np.float32),
                "tau": tau,
                "T": T,
            }
        )
    return patients


def make_labels(patients: Sequence[dict], cfg: Config) -> List[dict]:
    """Hourly target: event onset occurs in (t, t+horizon]."""
    for p in patients:
        T, tau = p["T"], p["tau"]
        y = np.zeros(T, dtype=np.int8)
        if tau is not None:
            end = min(T, tau)
            start = max(0, tau - cfg.horizon)
            y[start:end] = 1
        p["y"] = y
    return list(patients)


def _feat_window(X: np.ndarray, t: int) -> np.ndarray:
    cur = X[t]
    prev = X[t - 1] if t > 0 else X[t]
    hist = X[max(0, t - 3):t + 1].mean(axis=0)
    slope = cur - prev
    return np.concatenate([cur, slope, hist]).astype(np.float32)


class RiskForecaster:
    """
    Stage-1 forecaster.

    Fit the classifier on TRAIN only and fit isotonic calibration on VALIDATION.
    TEST is never used for fitting or calibration.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.clf = HistGradientBoostingClassifier(
            max_depth=3,
            max_iter=250,
            learning_rate=0.06,
            random_state=cfg.seed,
        )
        self.cal = IsotonicRegression(out_of_bounds="clip")
        self.fitted = False

    @staticmethod
    def _matrix(patients):
        F, Y = [], []
        for p in patients:
            for t in range(p["T"]):
                F.append(_feat_window(p["X"], t))
                Y.append(p["y"][t])
        return np.asarray(F, dtype=np.float32), np.asarray(Y, dtype=np.int8)

    def fit(self, train_patients, calibration_patients):
        F_train, y_train = self._matrix(train_patients)
        F_cal, y_cal = self._matrix(calibration_patients)

        self.clf.fit(F_train, y_train)
        raw_cal = self.clf.predict_proba(F_cal)[:, 1]

        # Isotonic calibration needs both classes.
        if np.unique(y_cal).size < 2:
            raise ValueError("Calibration split must contain both classes.")
        self.cal.fit(raw_cal, y_cal)
        self.fitted = True
        return self

    def risk_traj(self, patient):
        if not self.fitted:
            raise RuntimeError("RiskForecaster is not fitted.")
        F = np.asarray(
            [_feat_window(patient["X"], t) for t in range(patient["T"])],
            dtype=np.float32,
        )
        raw = self.clf.predict_proba(F)[:, 1]
        return self.cal.transform(raw).astype(np.float32)

    def evaluate(self, patients) -> Dict[str, float]:
        p = np.concatenate([self.risk_traj(x) for x in patients])
        y = np.concatenate([x["y"] for x in patients])
        return {
            "AUROC": float(roc_auc_score(y, p)),
            "AUPRC": float(average_precision_score(y, p)),
            "positive_rate": float(y.mean()),
        }


def in_window(tau: Optional[int], t: int, cfg: Config) -> bool:
    return tau is not None and (tau - cfg.horizon) <= t < tau


def p_act(rho: float, cfg: Config) -> float:
    return float(np.clip((rho - 0.20) / 0.55, cfg.act_floor, 1.0))


def clinician_acts(rho: float, rng: np.random.Generator, cfg: Config) -> bool:
    return bool(rng.random() < p_act(rho, cfg))


def state_from_risk(pr, t, T, rho=None):
    dp = float(pr[t] - (pr[t - 1] if t > 0 else pr[t]))
    base = [float(pr[t]), dp, float(t / max(T, 1))]
    if rho is not None:
        base.append(float(rho))
    return np.asarray(base, dtype=np.float32)


def state_dim(use_trust_state: bool) -> int:
    return 4 if use_trust_state else 3


def _reward_alarm(true_alarm, acted, tau, t, rho, cfg: Config, mode: str):
    lead = (tau - t) if tau is not None else 0

    if mode == "naive":
        r = cfg.r_hit if true_alarm else -cfg.r_fa
        new_rho = (
            min(1.0, rho + cfg.trust_rec_true)
            if true_alarm
            else max(0.0, rho - cfg.trust_dec_fa)
        )
        return r, new_rho

    if mode == "fixed":
        if true_alarm and 0 < lead <= cfg.horizon:
            r = cfg.r_hit * min(lead, cfg.horizon) / cfg.horizon
            new_rho = min(1.0, rho + cfg.trust_rec_true)
        else:
            r = -cfg.r_fa
            new_rho = max(0.0, rho - cfg.trust_dec_fa)
        return r, new_rho

    if mode == "trust":
        if true_alarm and 0 < lead <= cfg.horizon:
            base = cfg.r_hit * min(lead, cfg.horizon) / cfg.horizon
            # An ignored true alarm still has a small residual value, but much less
            # than an acted-upon alarm. This matches the supplied guide code.
            r = base * (1.0 if acted else cfg.ignored_true_fraction)
            new_rho = min(1.0, rho + cfg.trust_rec_true)
        else:
            # Derived from trust erosion rather than a constant false-alarm cost.
            r = -cfg.r_fa * (1.0 + cfg.kappa * rho)
            new_rho = max(0.0, rho - cfg.trust_dec_fa)
        return r, new_rho

    raise ValueError(f"Unknown reward mode: {mode}")


def _reward_terminal(tau, rho, cfg: Config, mode: str):
    if tau is not None:
        return -cfg.r_miss, rho
    return cfg.r_tn, min(1.0, rho + cfg.trust_rec)


def rollout_buffer(
    patients,
    forecaster: RiskForecaster,
    cfg: Config,
    mode: str,
    rng: np.random.Generator,
    explore: float = 0.25,
):
    """
    Build an offline transition buffer.

    mode:
      naive: per-step classification-style RL, no trust in state
      fixed_no_trust: absorbing alarm, fixed penalty, no trust in state
      fixed_with_trust: absorbing alarm, fixed penalty, trust in state
      trust_aware: absorbing alarm, trust in state + trust-derived reward
    """
    if mode == "naive":
        reward_mode, use_trust_state, absorbing = "naive", False, False
    elif mode == "fixed_no_trust":
        reward_mode, use_trust_state, absorbing = "fixed", False, True
    elif mode == "fixed_with_trust":
        reward_mode, use_trust_state, absorbing = "fixed", True, True
    elif mode == "trust_aware":
        reward_mode, use_trust_state, absorbing = "trust", True, True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    D = []
    rho = 1.0

    for p in patients:
        pr = forecaster.risk_traj(p)
        T, tau = p["T"], p["tau"]
        alarmed = False

        for t in range(T):
            s = state_from_risk(pr, t, T, rho if use_trust_state else None)
            true_now = in_window(tau, t, cfg)

            # Behavior policy with forced alarm exploration.
            if rng.random() < explore:
                a = ALARM if rng.random() < 0.15 else WAIT
            else:
                a = ALARM if pr[t] > 0.5 else WAIT

            if not absorbing:
                if a == ALARM:
                    r = cfg.r_hit if true_now else -cfg.r_fa
                    rho = (
                        min(1.0, rho + cfg.trust_rec_true)
                        if true_now
                        else max(0.0, rho - cfg.trust_dec_fa)
                    )
                else:
                    r = cfg.r_tn if not true_now else -cfg.r_miss / cfg.horizon

                done = t == T - 1
                if t + 1 < T:
                    s2 = state_from_risk(
                        pr, t + 1, T, rho if use_trust_state else None
                    )
                else:
                    s2 = np.zeros_like(s)

                D.append((s, a, r, s2, done))
                continue

            if a == ALARM:
                acted = clinician_acts(rho, rng, cfg)
                r, rho = _reward_alarm(
                    true_now, acted, tau, t, rho, cfg, reward_mode
                )
                D.append((s, a, r, np.zeros_like(s), True))
                alarmed = True
                break

            r = -cfg.wait_cost
            if t + 1 < T:
                s2 = state_from_risk(
                    pr, t + 1, T, rho if use_trust_state else None
                )
                D.append((s, a, r, s2, False))
            else:
                rterm, rho = _reward_terminal(tau, rho, cfg, reward_mode)
                D.append((s, a, r + rterm, np.zeros_like(s), True))

        if absorbing and not alarmed:
            rho = min(1.0, rho + cfg.trust_rec)

    return D


class FQIController:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.q = None
        self.use_trust_state = None

    def fit(self, transitions, log=False):
        if not transitions:
            raise ValueError("Empty offline transition buffer.")

        S = np.asarray([d[0] for d in transitions], dtype=np.float32)
        A = np.asarray([d[1] for d in transitions], dtype=np.int8)
        R = np.asarray([d[2] for d in transitions], dtype=np.float32)
        S2 = np.asarray([d[3] for d in transitions], dtype=np.float32)
        done = np.asarray([d[4] for d in transitions], dtype=bool)

        SA = np.hstack([S, A[:, None].astype(np.float32)])
        Y = R.copy()
        self.use_trust_state = S.shape[1] == 4
        self.alarm_rate_log = []

        for _ in range(self.cfg.fqi_iters):
            self.q = ExtraTreesRegressor(
                n_estimators=self.cfg.n_trees,
                min_samples_leaf=self.cfg.min_samples_leaf,
                n_jobs=1,
                random_state=self.cfg.seed,
            )
            self.q.fit(SA, Y)

            sw = np.hstack([S2, np.full((len(S2), 1), WAIT, dtype=np.float32)])
            sa = np.hstack([S2, np.full((len(S2), 1), ALARM, dtype=np.float32)])
            qw = self.q.predict(sw)
            qa = self.q.predict(sa)

            Y = R + self.cfg.gamma * (~done) * np.maximum(qw, qa)
            if log:
                self.alarm_rate_log.append(self.greedy_alarm_rate(S))

        return self

    def predict_q(self, S):
        if self.q is None:
            raise RuntimeError("Controller is not fitted.")
        S = np.atleast_2d(S).astype(np.float32)
        qw = self.q.predict(
            np.hstack([S, np.full((len(S), 1), WAIT, dtype=np.float32)])
        )
        qa = self.q.predict(
            np.hstack([S, np.full((len(S), 1), ALARM, dtype=np.float32)])
        )
        return qw, qa

    def act(self, s):
        qw, qa = self.predict_q(s)
        return ALARM if qa[0] > qw[0] else WAIT

    def greedy_alarm_rate(self, S):
        qw, qa = self.predict_q(S)
        return float(np.mean(qa > qw))


def eval_policy(
    patients,
    forecaster,
    cfg,
    rng,
    policy="fqi",
    controller=None,
    threshold=0.5,
    use_trust_state=True,
    sample_clinician=True,
):
    """
    Evaluate one alarm policy over a patient stream.

    Metrics:
      event_sens_raised: event detected by an alarm
      event_sens_acted: event detected by an alarm that the clinician acts upon
      median_lead: median lead time among acted-upon detections
      false_alarms_per_100h
      alarms_per_100h
      mean_trust
    """
    rho = 1.0
    raised = acted_detect = events = fa = alarms = total_hours = 0
    leads = []
    trust_trace = []

    for p in patients:
        pr = forecaster.risk_traj(p)
        T, tau = p["T"], p["tau"]
        events += int(tau is not None)
        total_hours += T
        fired_t = None

        for t in range(T):
            if policy == "threshold":
                a = ALARM if pr[t] >= threshold else WAIT
            else:
                s = state_from_risk(
                    pr, t, T, rho if use_trust_state else None
                )
                a = controller.act(s)

            if a == ALARM:
                fired_t = t
                break

        if fired_t is None:
            rho = min(1.0, rho + cfg.trust_rec)
            trust_trace.append(rho)
            continue

        alarms += 1
        true_a = in_window(tau, fired_t, cfg)

        if sample_clinician:
            acted = clinician_acts(rho, rng, cfg)
        else:
            acted = p_act(rho, cfg)  # expected action probability

        if true_a:
            raised += 1
            # For stochastic evaluation, acted is bool; for expected evaluation
            # it is a probability.
            if sample_clinician:
                if acted:
                    acted_detect += 1
                    leads.append(tau - fired_t)
            else:
                acted_detect += float(acted)
                leads.extend([tau - fired_t])
            rho = min(1.0, rho + cfg.trust_rec_true)
        else:
            fa += 1
            rho = max(0.0, rho - cfg.trust_dec_fa)

        trust_trace.append(rho)

    denom = max(1, events)
    return {
        "event_sens_raised": float(raised / denom),
        "event_sens_acted": float(acted_detect / denom),
        "median_lead": float(np.median(leads)) if leads else 0.0,
        "false_alarms_per_100h": float(100.0 * fa / max(1, total_hours)),
        "alarms_per_100h": float(100.0 * alarms / max(1, total_hours)),
        "mean_trust": float(np.mean(trust_trace)) if trust_trace else 1.0,
        "events": int(events),
        "total_hours": int(total_hours),
    }


def choose_threshold_by_burden(
    patients,
    forecaster,
    cfg,
    rng,
    target_burden,
    thresholds=None,
):
    """Choose threshold on VALIDATION only."""
    thresholds = (
        np.linspace(0.05, 0.95, 37)
        if thresholds is None
        else np.asarray(thresholds)
    )

    rows = []
    for thr in thresholds:
        m = eval_policy(
            patients,
            forecaster,
            cfg,
            rng,
            policy="threshold",
            threshold=float(thr),
            use_trust_state=False,
        )
        rows.append((float(thr), m))

    best_thr, best_m = min(
        rows,
        key=lambda x: abs(x[1]["alarms_per_100h"] - target_burden),
    )
    return best_thr, best_m, rows



def build_tabular_transitions(
    df,
    cfg: Config,
    mode: str = "trust_aware",
    behavior_threshold: float = 0.5,
    explore: float = 0.05,
    seed: int = 0,
):
    """Build FQI transitions from an hourly risk table.

    This is appropriate for a retrospective *alarm-policy simulation* when actual
    historical alarm actions are unavailable. It must not be described as logged
    clinician actions. The behavior policy is explicit and auditable.

    Required columns: subject_id, stay_id, hour, risk, sae_onset_hour.
    """
    required = {"subject_id", "stay_id", "hour", "risk", "sae_onset_hour"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for transition building: {sorted(missing)}")

    if mode == "fixed_no_trust":
        use_trust_state = False
        reward_mode = "fixed"
        absorbing = True
    elif mode == "fixed_with_trust":
        use_trust_state = True
        reward_mode = "fixed"
        absorbing = True
    elif mode == "trust_aware":
        use_trust_state = True
        reward_mode = "trust"
        absorbing = True
    else:
        raise ValueError("Tabular transition builder supports fixed_no_trust, fixed_with_trust, trust_aware")

    rng = np.random.default_rng(seed)
    D = []
    rho = 1.0

    df = df.sort_values(["subject_id", "stay_id", "hour"]).copy()
    for (_, _), g in df.groupby(["subject_id", "stay_id"], sort=False):
        g = g.reset_index(drop=True)
        T = len(g)
        alarmed = False
        for i, row in g.iterrows():
            risk = float(row["risk"])
            prev = float(g.loc[i - 1, "risk"]) if i > 0 else risk
            t_frac = i / max(T, 1)
            s = np.array([risk, risk - prev, t_frac] + ([rho] if use_trust_state else []), dtype=np.float32)

            if rng.random() < explore:
                a = int(rng.random() < 0.15)
            else:
                a = int(risk >= behavior_threshold)

            onset = row["sae_onset_hour"]
            tau = None if pd.isna(onset) else float(onset)
            hour = float(row["hour"])
            true_now = tau is not None and (tau - cfg.horizon) <= hour < tau

            if a == ALARM:
                acted = clinician_acts(rho, rng, cfg)
                r, rho = _reward_alarm(true_now, acted, tau, hour, rho, cfg, reward_mode)
                D.append((s, a, r, np.zeros_like(s), True))
                alarmed = True
                break

            if i + 1 < T:
                next_row = g.loc[i + 1]
                next_risk = float(next_row["risk"])
                next_prev = risk
                s2 = np.array(
                    [next_risk, next_risk - next_prev, (i + 1) / max(T, 1)]
                    + ([rho] if use_trust_state else []),
                    dtype=np.float32,
                )
                D.append((s, WAIT, -cfg.wait_cost, s2, False))
            else:
                rterm, rho = _reward_terminal(tau, rho, cfg, reward_mode)
                D.append((s, WAIT, rterm - cfg.wait_cost, np.zeros_like(s), True))

        if not alarmed:
            rho = min(1.0, rho + cfg.trust_rec)

    return D

def evaluate_all(
    train,
    val,
    test,
    forecaster,
    cfg,
    seed=0,
):
    """Train the three/four controller variants and evaluate on held-out TEST."""
    rng_train = np.random.default_rng(10_000 + seed)
    rng_val = np.random.default_rng(20_000 + seed)
    rng_test = np.random.default_rng(30_000 + seed)

    modes = [
        "naive",
        "fixed_no_trust",
        "fixed_with_trust",
        "trust_aware",
    ]
    results = []
    controllers = {}

    for mode in modes:
        D = rollout_buffer(train, forecaster, cfg, mode, rng_train)
        ctrl = FQIController(cfg).fit(D)
        controllers[mode] = ctrl
        m = eval_policy(
            test,
            forecaster,
            cfg,
            rng_test,
            policy="fqi",
            controller=ctrl,
            use_trust_state=(mode in ("fixed_with_trust", "trust_aware")),
        )
        m.update({"method": mode, "seed": seed})
        results.append(m)

    trust_burden = results[-1]["alarms_per_100h"]

    # Threshold is selected on VALIDATION, then frozen and evaluated on TEST.
    threshold, val_m, _ = choose_threshold_by_burden(
        val, forecaster, cfg, rng_val, trust_burden
    )
    test_m = eval_policy(
        test,
        forecaster,
        cfg,
        rng_test,
        policy="threshold",
        threshold=threshold,
        use_trust_state=False,
    )
    test_m.update(
        {
            "method": "threshold_matched",
            "seed": seed,
            "threshold": threshold,
            "validation_alarm_burden": val_m["alarms_per_100h"],
        }
    )
    results.append(test_m)

    return results, controllers


def split_patients(patients, train_frac=0.60, val_frac=0.20, seed=0):
    """Patient-level split; no patient appears in more than one split."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(patients))
    n = len(idx)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = [patients[i] for i in idx[:n_train]]
    val = [patients[i] for i in idx[n_train:n_train + n_val]]
    test = [patients[i] for i in idx[n_train + n_val:]]
    return train, val, test
