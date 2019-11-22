from typing import Dict, List

import numpy as np

ObsType = Dict[str, np.ndarray]


def sanitize_observation(obs: ObsType, num_traffic_lights=5, num_edges=4) -> ObsType:
    """ Sanitizes MultiTrafficLightGridPOEnv observations to only include density and average
    velocities of cars

    @param obs the default observation
    @param num_traffic_lights the number of traffic lights in the observation space
        (see MultiTrafficLightGridPOEnv)
    @param num_edges the number of edges in the observation space (see MultiTrafficLightGridPOEnv)
    """
    return {
        k: obs[k][
            -1 * (2 * num_edges + 3 * num_traffic_lights) : -1 * 3 * num_traffic_lights
        ]
        for k in obs
    }


def generate_graph(sanitized_obs) -> Dict[str, List[str]]:
    """ Generates an interaction graph based on the densities and velocitiies of cars
    @param sanitized_obs the output of sanitize_observation
    """
    raise NotImplementedError


# TODO(williammacke): Decide if you want a single neighbor weight or something more sophisticated.
def calculate_adjusted_rewards(
    raw_rewards: Dict[str, float],
    interaction_graph: Dict[str, List[str]],
    neigbhor_weight: float,
) -> Dict[str, float]:
    """ Adjusts the raw reward to include the rewards of your neighbors.
    
    @param raw_rewards dict mapping from an agent's name to its base reward this timestep
    @param interaction_graph mapping from an agent's name to a list of its neighbors
    @param neighbor_weight how much to weight your neighbor's rewards relative to your own
    """
    raise NotADirectoryError


def calculate_expected_return(
    interaction_graph: Dict[str, List[str]],
    sp: ObsType,
    rewards: Dict[str, float],
    values: Dict[str, float],
    gamma=0.9,
) -> Dict[str, float]:
    """ Calculates the per agent expected return using TD(0)
    @param interaction_graph the global interaction graph
    @param sp dictionary mapping agents to resulting states
    @param rewards dictionary mapping agents to rewards received
    @param values dictionary mapping agents to current value functions
    @param gamma the discount factor
    """
    expected_returns = {}

    for agent in interaction_graph:
        count = 0
        total_value = 0
        for neighbor in interaction_graph[agent]:
            total_value += values[neighbor](sp[neighbor])
            count += 1
        mean_value = total_value / count
        expected_returns[agent] = rewards[agent] + gamma * mean_value
    return expected_returns
