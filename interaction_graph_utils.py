

"""
Sanitizes observation to only include density and average velocities of
cars in the network grid
@param obs the default observation
@param num_traffic_lights the number of traffic lights in the observation space (see multiagent_traffic_light_grid env)
@param num_edges the number of edges in the observation space (see multiagent_traffic_light_grid env)
"""
def sanitizeObservation(obs, num_traffic_lights=4, num_edges=4):
    raise NotImplementedError

"""
generates an interaction graph based on the densities and velocitiies of cars in the network grid
@param sanitizedObs the output of sanitizedObservation
"""
def generateGraph(sanitizedObs):
    raise NotImplementedError


"""
calculate the per agent expected return
@param IG the global interaction graph
@param sp dictionary mapping agents to resulting states
@param rewards dictionary mapping agents to rewards received
@param values dictionary mapping agents to current value functions
@param gamma the discount factor
"""
def calculateExpectedReturn(IG, sp, rewards, values, gamma=0.9):
    raise NotImplementedError
