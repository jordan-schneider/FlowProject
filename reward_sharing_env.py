from typing import Dict

from flow.envs.multiagent.traffic_light_grid import MultiTrafficLightGridPOEnv
import numpy as np


class RewardSharingEnv(MultiTrafficLightGridPOEnv):
    """ Multiagent traffic light grid environment with reward sharing """

    def __init__(self, env_params, sim_params, network, simulator="traci"):
        super().__init__(env_params, sim_params, network, simulator)
        self.neighbor_weight: float = env_params.additional_params["neighbor_weight"]
        assert isinstance(
            self.neighbor_weight, float
        ), "Neighbor weight must be a float."
        self.raw_reward = 0

    def compute_reward(self, rl_actions, **kwargs):
        """ Adjusts the raw reward to include the rewards of your neighbors.

        @returns a mapping from an agent's name to its sharing adjusted reward
        """
        raw_rewards: Dict[str, float] = super().compute_reward(rl_actions, **kwargs)

        adjusted_rewards: Dict[str, float] = dict()
        self.raw_reward += np.sum([raw_rewards[k] for k in raw_rewards])

        directions = self.direction.flatten()
        for rl_id in raw_rewards.keys():
            adjusted_rewards[rl_id] = raw_rewards[rl_id]

            rl_id_num = self.__get_id_num_from_name(rl_id)
            direction = directions[rl_id_num]
            if direction == 0:
                top = self.__get_name_from_id(self._get_relative_node(rl_id, "top"))
                bottom = self.__get_name_from_id(
                    self._get_relative_node(rl_id, "bottom")
                )
                if top != "center-1":
                    adjusted_rewards[rl_id] += raw_rewards[top] * self.neighbor_weight
                if bottom != "center-1":
                    adjusted_rewards[rl_id] += (
                        raw_rewards[bottom] * self.neighbor_weight
                    )
            else:
                left = self.__get_name_from_id(self._get_relative_node(rl_id, "left"))
                right = self.__get_name_from_id(self._get_relative_node(rl_id, "right"))
                if left != "center-1":
                    adjusted_rewards[rl_id] += raw_rewards[left] * self.neighbor_weight
                if right != "center-1":
                    adjusted_rewards[rl_id] += raw_rewards[right] * self.neighbor_weight
        return adjusted_rewards

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
