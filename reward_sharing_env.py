from typing import Dict

from flow.flow.envs.multiagent.traffic_light_grid import MultiTrafficLightGridPOEnv


class RewardSharingEnv(MultiTrafficLightGridPOEnv):
    """ Multiagent traffic light grid environment with reward sharing """

    def __init__(self, neighbor_weight: float, **kwargs):
        super().__init__(**kwargs)
        self.neighbor_weight = neighbor_weight

    def compute_reward(self, rl_actions, **kwargs):
        """ Adjusts the raw reward to include the rewards of your neighbors.

        @returns a mapping from an agent's name to its sharing adjusted reward
        """
        raw_rewards: Dict[str, float] = self.super(rl_actions, **kwargs)

        adjusted_rewards: Dict[str, float] = dict()

        directions = self.direction.flatten()
        for rl_id in raw_rewards.keys():
            adjusted_rewards[rl_id] = raw_rewards[rl_id]

            rl_id_num = self.__get_id_num_from_name(rl_id)
            direction = directions[rl_id_num]
            if direction == 0:
                top = self._get_relative_node(rl_id, "top")
                bottom = self._get_relative_node(rl_id, "bottom")
                if top > 0:
                    adjusted_rewards[rl_id] += raw_rewards[top] * self.neighbor_weight
                if bottom > 0:
                    adjusted_rewards[rl_id] += (
                        raw_rewards[bottom] * self.neighbor_weight
                    )
            else:
                left = self._get_relative_node(rl_id, "left")
                right = self._get_relative_node(rl_id, "right")
                if left > 0:
                    adjusted_rewards[rl_id] += raw_rewards[left] * self.neighbor_weight
                if right > 0:
                    adjusted_rewards[rl_id] += raw_rewards[right] * self.neighbor_weight
        return adjusted_rewards

    @staticmethod
    def __get_id_num_from_name(name: str) -> int:
        return int(name.split("center")[1])
