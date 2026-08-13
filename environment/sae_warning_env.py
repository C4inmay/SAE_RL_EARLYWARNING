import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class SAEWarningEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, dataframe):

        super().__init__()

        self.df = dataframe.copy()

        # RL state
        self.state_columns = [
            "gcs_last_observed",
            "previous_observed_gcs",
            "heart_rate",
            "map",
            "resp_rate",
            "spo2",
            "hour"
        ]

        required_columns = [
            "subject_id",
            "hadm_id",
            "icustay_id",
            "hour",
            "future_sae_1h",
            *self.state_columns
        ]

        missing = [
            col for col in required_columns
            if col not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        # Keep chronological order
        self.df = self.df.sort_values(
            ["icustay_id", "hour"]
        ).reset_index(drop=True)

        # One ICU stay = one episode
        self.episodes = {
            stay_id: group.reset_index(drop=True)
            for stay_id, group in self.df.groupby("icustay_id")
        }

        self.stay_ids = list(self.episodes.keys())

        # 3 warning actions
        self.action_space = spaces.Discrete(3)

        # 7-dimensional state
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.state_columns),),
            dtype=np.float32
        )

        self.current_episode = None
        self.current_stay_id = None
        self.current_step = 0

    def _get_state(self):

        row = self.current_episode.iloc[
            self.current_step
        ]

        state = []

        for column in self.state_columns:

            value = row[column]

            if pd.isna(value):
                value = 0.0

            state.append(float(value))

        return np.asarray(
            state,
            dtype=np.float32
        )

    def _get_reward(self, action, future_sae):

       if future_sae == 0:

                if action == 0:
                    return 1.0

                elif action == 1:
                    return -0.5

                elif action == 2:
                    return -2.0
       else:

                if action == 0:
                    return -5.0

                elif action == 1:
                    return 3.0

                elif action == 2:
                    return 5.0

                    return 0.0

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        # Random ICU stay
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
                self.current_episode.iloc[0]["hour"]
            )
        }

        return state, info

    def step(self, action):

        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action: {action}"
            )

        current_row = self.current_episode.iloc[
            self.current_step
        ]

        sae = int(current_row["sae"])

        future_sae = int(
    current_row["future_sae_1h"]
)

        reward = self._get_reward(
            action,
            future_sae
        )   

        self.current_step += 1

        terminated = (
            self.current_step >=
            len(self.current_episode)
        )

        truncated = False

        if terminated:

            next_state = np.zeros(
                len(self.state_columns),
                dtype=np.float32
            )

        else:

            next_state = self._get_state()

        info = {
       "icustay_id": self.current_stay_id,
       "hour": int(current_row["hour"]),
       "future_sae": future_sae,
       "action": action
}

        return (
            next_state,
            reward,
            terminated,
            truncated,
            info
        )

    def render(self):

        print(
            f"ICU stay: {self.current_stay_id} | "
            f"Step: {self.current_step}"
        )