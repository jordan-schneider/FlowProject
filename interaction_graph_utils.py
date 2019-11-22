

"""
Sanitizes observation to only include density and average velocities of
cars in the network grid
@param obs the default observation
@param num_traffic_lights the number of traffic lights in the observation space (see multiagent_traffic_light_grid env)
@param num_edges the number of edges in the observation space (see multiagent_traffic_light_grid env)
"""
def sanitize_observation(obs, num_traffic_lights=5, num_edges=4):
    return {k:obs[k][-1*(2*num_edges+3*num_traffic_lights):-1*3*num_traffic_lights] for k in obs}

"""
generates an interaction graph based on the densities and velocitiies of cars in the network grid
@param sanitizedObs the output of sanitizedObservation
"""
def generate_graph(sanitizedObs):
    raise NotImplementedError


"""
calculate the per agent expected return using TD(0)
@param IG the global interaction graph
@param sp dictionary mapping agents to resulting states
@param rewards dictionary mapping agents to rewards received
@param values dictionary mapping agents to current value functions
@param gamma the discount factor
"""
def calculate_expected_return(IG, sp, rewards, values, gamma=0.9):
    expected_returns = {}
    for agent in IG:
        count = 0
        totalVal = 0
        for neighbor in IG[agent]:
            totalVal += values[neighbor](sp[neighbor])
            count += 1
        avgVal = totalVal / count
        expected_returns[agent] = rewards[agent]+gamma*avgVal
    return expected_returns

