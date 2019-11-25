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

def generate_graph_from_env(env):
    """
    generates the interaction graph from a MultiTrafficLightGridPOEnv
    @param env the environment
    """
    directions = env.direction.flatten()
    num_ids = len(env.k.traffic_light.get_ids())
    graph = np.zeros(shape=(num_ids, num_ids), dtype=int)
    for rl_id in env.k.traffic_light.get_ids():
        rl_id_num = int(rl_id.split("center")[1])
        direction = directions[rl_id_num]
        if direction == 0:
            top =  env._get_relative_node(rl_id, "top")
            bottom = env._get_relative_node(rl_id, "bottom")
            if top > 0: 
                graph[rl_id_num][top] = 1
            if bottom > 0: 
                graph[rl_id_num][bottom] = 1
        else:
            left = env._get_relative_node(rl_id, "left")
            right = env._get_relative_node(rl_id, "right")
            if left > 0:
                graph[rl_id_num][left] = 1
            if right > 0:
                graph[rl_id_num][right] = 1
    return graph

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
