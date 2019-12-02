from typing import Dict
import operator
import math

from flow.envs.multiagent.traffic_light_grid import MultiTrafficLightGridPOEnv


class RewardSharingEnv(MultiTrafficLightGridPOEnv):
    """ Multiagent traffic light grid environment with Colight reward sharing pattern """

    def __init__(self, env_params, sim_params, network, simulator="traci"):
        super().__init__(env_params, sim_params, network, simulator)
        self.k_nearest_neighbor: int = env_params.additional_params["k_nearest_neighbor"]
        assert isinstance(
            self.k_nearest_neighbor, int
        ), "Neighbor count k must be an integer."

        self.temperature_factor: float = env_params.additional_params["temperature_factor"]
        assert isinstance(
            self.temperature_factor, float
        ), "Temperature factor must be a float."

    def compute_reward(self, rl_actions, **kwargs):
        """ Adjusts the raw reward to include the rewards of not only neighbors, but also intersections with certain distrance from target agent.

        @returns a mapping from an agent's name to its sharing adjusted reward
        """
        raw_rewards: Dict[str, float] = super().compute_reward(rl_actions, **kwargs)
        adjusted_rewards: Dict[str, float] = dict()

        for rl_id in raw_rewards.keys():
            adjusted_rewards[rl_id] = 0
            rl_id_num = self.__get_id_num_from_name(rl_id)
            kNN_importance = self.get_kNN_importance(rl_id_num, raw_rewards, self.k_nearest_neighbor, self.temperature_factor)

            for neighbor_id in kNN_importance.keys():
                adjusted_rewards[rl_id] += raw_rewards[neighbor_id] * kNN_importance[neighbor_id]

        return adjusted_rewards

    @staticmethod
    def __get_id_num_from_name(name: str) -> int:
        return int(name.split("center")[1])

    @staticmethod
    def __get_name_from_id(rl_id: int) -> str:
        return f"center{rl_id}"

    def get_distance_from_id(self, rl_id_source, rl_id_target):
        row_source = rl_id_source // self.cols
        row_target = rl_id_target // self.cols
        y_dist = abs(row_source - row_target)

        col_source = rl_id_source % self.cols
        col_target = rl_id_target % self.cols
        x_dist = abs(col_source - col_target)

        return x_dist + y_dist

    def get_kNN_importance(self, rl_id_num, raw_rewards, k, tau):
        kNN_importance: Dict[str, float] = dict()

        for rl_id in raw_rewards.keys():
            rl_id_source = self.__get_id_num_from_name(rl_id)
            distance = self.get_distance_from_id(rl_id_source, rl_id_num)
            if len(kNN_importance) == k:
                min_importance_id = min(kNN_importance.items(), key=operator.itemgetter(1))[0]
                if -distance > kNN_importance[min_importance_id]:
                    kNN_importance[rl_id] = -distance
                    del kNN_importance[min_importance_id]
            else:
                kNN_importance[rl_id] = -distance

        softmax_normalizer = 0
        for neighbor_id in kNN_importance.keys():
            softmax_normalizer += math.exp(kNN_importance[neighbor_id]/tau)

        for neighbor_id in kNN_importance.keys():
            kNN_importance[neighbor_id] = math.exp(kNN_importance[neighbor_id]/tau) / softmax_normalizer

        return kNN_importance