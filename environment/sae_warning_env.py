import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class SAEWarningEnv(gym.Env):
    """
    Offline temporal RL environment for early warning
    of neurological deterioration associated with SAE.

    One ICU stay is treated as one episode.

    Actions:
        0 = Routine monitoring
        1 = Increased surveillance
        2 = Clinical escalation

    State:
        [GCS, GCS_change, MAP, Heart Rate,
         Respiratory Rate, SpO2, ICU hour]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, dataframe):

        super().__init__()

        self.df = dataframe.copy()

        # --------------------------------------------------
        # Required columns
        # --------------------------------------------------

        self.state_columns = [
            "gcs_total",
            "gcs_change",
            "map",
            "heart_rate",
            "resp_rate",
            "spo2",
            "hour",
        ]

        required_columns = [
            "subject_id",
            "hadm_id",
            "icustay_id",
            "hour",
            *self.state_columns,
            "deterioration",
        ]

        missing = [
            col for col in required_columns
            if col not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        # --------------------------------------------------
        # Sort temporal data
        # --------------------------------------------------

        self.df = self.df.sort_values(
            ["icustay_id", "hour"]
        ).reset_index(drop=True)

        # --------------------------------------------------
        # Group ICU stays into episodes
        # --------------------------------------------------

        self.episodes = {
            stay_id: group.reset_index(drop=True)
            for stay_id, group
            in self.df.groupby("icustay_id")
        }

        self.stay_ids = list(self.episodes.keys())

        # Current episode
        self.current_episode = None
        self.current_stay_id = None
        self.current_step = 0

        # --------------------------------------------------
        # Action space
        # --------------------------------------------------

        self.action_space = spaces.Discrete(3)

        # --------------------------------------------------
        # State space
        #
        # We use normalized/clipped continuous values.
        # --------------------------------------------------

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(len(self.state_columns),),
            dtype=np.float32,
        )

    # ======================================================
    # STATE
    # ======================================================

    def _get_state(self):

        row = self.current_episode.iloc[self.current_step]

        values = []

        for column in self.state_columns:

            value = row[column]

            if pd.isna(value):
                value = 0.0

            values.append(float(value))

        state = np.asarray(
            values,
            dtype=np.float32
        )

        return state

    # ======================================================
    # REWARD
    # ======================================================

    def _calculate_reward(self, action, deterioration):

        # ----------------------------------------------
        # No deterioration
        # ----------------------------------------------

        if deterioration == 0:

            if action == 0:
                return 1.0

            elif action == 1:
                return -0.5

            elif action == 2:
                return -2.0

        # ----------------------------------------------
        # Deterioration
        # ----------------------------------------------

        else:

            if action == 0:
                return -5.0

            elif action == 1:
                return 3.0

            elif action == 2:
                return 5.0

        return 0.0

    # ======================================================
    # RESET
    # ======================================================

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        # Select an ICU stay
        self.current_stay_id = self.np_random.choice(
            self.stay_ids
        )

        self.current_episode = self.episodes[
            self.current_stay_id
        ]

        self.current_step = 0

        state = self._get_state()

        info = {
            "icustay_id": self.current_stay_id,
            "hour": int(
                self.current_episode.iloc[
                    self.current_step
                ]["hour"]
            ),
        }

        return state, info

    # ======================================================
    # STEP
    # ======================================================

    def step(self, action):

        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action: {action}"
            )

        current_row = self.current_episode.iloc[
            self.current_step
        ]

        deterioration = int(
            current_row["deterioration"]
        )

        reward = self._calculate_reward(
            action,
            deterioration
        )

        # Move forward one hour
        self.current_step += 1

        terminated = (
            self.current_step
            >= len(self.current_episode)
        )

        truncated = False

        if terminated:

            # Return final state
            next_state = self._get_state()

        else:

            next_state = self._get_state()

        info = {
            "icustay_id": self.current_stay_id,
            "hour": int(current_row["hour"]),
            "deterioration": deterioration,
            "action": action,
        }

        return (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        )

    # ======================================================
    # RENDER
    # ======================================================

    def render(self):

        print(
            f"ICU stay: {self.current_stay_id} | "
            f"Step: {self.current_step}"
        )