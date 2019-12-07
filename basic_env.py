from typing import Dict
import tensorflow as tf
import numpy as np
from datetime import datetime

from flow.envs.multiagent.traffic_light_grid import MultiTrafficLightGridPOEnv


class BasicEnv(MultiTrafficLightGridPOEnv):
    """ Multiagent traffic light grid environment with reward sharing """

    def __init__(self, env_params, sim_params, network, simulator="traci"):
        super().__init__(env_params, sim_params, network, simulator)
        self.raw_reward = 0

    def compute_reward(self, rl_actions, **kwargs): 
        """ Adjusts the raw reward to include the rewards of your neighbors.

        @returns a mapping from an agent's name to its sharing adjusted reward
        """
        raw_rewards: Dict[str, float] = super().compute_reward(rl_actions, **kwargs)
        self.raw_reward += np.sum([raw_rewards[k] for k in raw_rewards])
        return raw_rewards


    @staticmethod
    def __get_id_num_from_name(name: str) -> int:
        return int(name.split("center")[1])

    @staticmethod
    def __get_name_from_id(rl_id: int) -> str:
        return f"center{rl_id}"
    def reset(self):
        obs = super().reset()
        self.raw_reward = 0
        return obs

