from typing import Dict

from flow.flow.envs.multiagent.traffic_light_grid import MultiTrafficLightGridPOEnv
from interaction_graph_utils import (
    calculate_adjusted_rewards,
    generate_graph,
    sanitize_observation,
)


class RewardSharingEnv(MultiTrafficLightGridPOEnv):
    """ Multiagent traffic light grid environment with reward sharing """

    def __init__(self, neighbor_weight: float, **kwargs):
        super().__init__(**kwargs)
        self.neighbor_weight = neighbor_weight

    def compute_reward(self, rl_actions, **kwargs):
        """ Computes reward for each agent by adjusting base reward to consider neighbor's rewards"""
        raw_rewards: Dict[str, float] = self.super(rl_actions, **kwargs)
        interaction_graph = generate_graph(
            sanitize_observation(
                obs=self.get_state(),
                num_edges=self.num_local_edges,
                num_traffic_lights=self.num_local_lights,
            )
        )
        return calculate_adjusted_rewards(
            raw_rewards=raw_rewards,
            interaction_graph=interaction_graph,
            neigbhor_weight=self.neigbhor_weight,
        )
