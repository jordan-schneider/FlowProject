"""Multi-agent traffic light example (single shared policy)."""

import argparse
import json

import ray
from ray import tune
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.controllers import GridRouter, SimCarFollowingController
from flow.core.params import (EnvParams, InFlows, InitialConfig, NetParams,
                              SumoCarFollowingParams, SumoParams,
                              VehicleParams)
from flow.networks import TrafficLightGridNetwork
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from reward_sharing_env import RewardSharingEnv
from reward_sharing_env_neighborhoods import RewardSharingEnvSimple
from reward_sharing_env_knn import RewardSharingEnvKNN
from flow.envs.multiagent.traffic_light_grid import MultiTrafficLightGridPOEnv
from basic_env import BasicEnv

try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class


# Experiment parameters
N_ROLLOUTS = 8  # number of rollouts per training iteration
N_CPUS = 8  # number of parallel workers

# Environment parameters
HORIZON = 400  # time horizon of a single rollout
V_ENTER = 30  # enter speed for departing vehicles
INNER_LENGTH = 300  # length of inner edges in the traffic light grid network
LONG_LENGTH = 100  # length of final edge in route
SHORT_LENGTH = 300  # length of edges that vehicles start on
# number of vehicles originating in the left, right, top, and bottom edges
N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 1, 1, 1, 1


def make_flow_params(n_rows, n_columns, edge_inflow, exp_env=BasicEnv, colight_k_nearest_neighbords=5, colight_temperature=0.5, neighbor_weight=0.1):
    """
    Generate the flow params for the experiment.

    Parameters
    ----------
    n_rows : int
        number of rows in the traffic light grid
    n_columns : int
        number of columns in the traffic light grid
    edge_inflow : float


    Returns
    -------
    dict
        flow_params object
    """
    # we place a sufficient number of vehicles to ensure they confirm with the
    # total number specified above. We also use a "right_of_way" speed mode to
    # support traffic light compliance
    vehicles = VehicleParams()
    num_vehicles = (N_LEFT + N_RIGHT) * n_columns + (N_BOTTOM + N_TOP) * n_rows
    vehicles.add(
        veh_id="human",
        acceleration_controller=(SimCarFollowingController, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            max_speed=V_ENTER,
            decel=7.5,  # avoid collisions at emergency stops
            speed_mode="right_of_way",
        ),
        routing_controller=(GridRouter, {}),
        num_vehicles=num_vehicles,
    )

    # inflows of vehicles are place on all outer edges (listed here)
    outer_edges = []
    outer_edges += ["left{}_{}".format(n_rows, i) for i in range(n_columns)]
    outer_edges += ["right0_{}".format(i) for i in range(n_rows)]
    outer_edges += ["bot{}_0".format(i) for i in range(n_rows)]
    outer_edges += ["top{}_{}".format(i, n_columns) for i in range(n_rows)]

    # equal inflows for each edge (as dictate by the EDGE_INFLOW constant)
    inflow = InFlows()
    for edge in outer_edges:
        inflow.add(
            veh_type="human",
            edge=edge,
            vehs_per_hour=edge_inflow,
            departLane="free",
            departSpeed=V_ENTER,
        )

    flow_params = dict(
        # name of the experiment
        exp_tag="grid_0_{}x{}_i{}_multiagent".format(n_rows, n_columns, edge_inflow),
        # name of the flow environment the experiment is running on
        env_name=exp_env,
        # name of the network class the experiment is running on
        network=TrafficLightGridNetwork,
        # simulator that is used by the experiment
        simulator="traci",
        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(restart_instance=True, sim_step=1, render=False,),
        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=HORIZON,
            additional_params={
                "target_velocity": 50,
                "switch_time": 3,
                "num_observed": 2,
                "discrete": False,
                "tl_type": "static",
                "num_local_edges": 4,
                "num_local_lights": 4,
                "neighbor_weight": neighbor_weight,
                "k_nearest_neighbor": colight_k_nearest_neighbords,
                "temperature_factor": colight_temperature
            },
        ),
        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            inflows=inflow,
            additional_params={
                "speed_limit": V_ENTER + 5,  # inherited from grid0 benchmark
                "grid_array": {
                    "short_length": SHORT_LENGTH,
                    "inner_length": INNER_LENGTH,
                    "long_length": LONG_LENGTH,
                    "row_num": n_rows,
                    "col_num": n_columns,
                    "cars_left": N_LEFT,
                    "cars_right": N_RIGHT,
                    "cars_top": N_TOP,
                    "cars_bot": N_BOTTOM,
                },
                "horizontal_lanes": 1,
                "vertical_lanes": 1,
            },
        ),
        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,
        # parameters specifying the positioning of vehicles upon initialization
        # or reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(spacing="custom", shuffle=True,),
    )
    return flow_params

def on_episode_end(info):
    env = info['env'].get_unwrapped()[0]
    episode = info['episode']
    episode.custom_metrics['raw_reward'] = env.raw_reward

def setup_exps_PPO(flow_params):
    """
    Experiment setup with PPO using RLlib.

    Parameters
    ----------
    flow_params : dictionary of flow parameters

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    alg_run = "PPO"
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config["num_workers"] = min(N_CPUS, N_ROLLOUTS)
    config["train_batch_size"] = HORIZON * N_ROLLOUTS
    config["simple_optimizer"] = True
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [32, 32]})
    config["lr"] = tune.grid_search([1e-4])
    config["horizon"] = HORIZON
    config["clip_actions"] = False  # FIXME(ev) temporary ray bug
    config["observation_filter"] = "NoFilter"
    config["callbacks"] = {'on_episode_end':tune.function(on_episode_end)}

    # save the flow params for replay
    flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config["env_config"]["flow_params"] = flow_json
    config["env_config"]["run"] = alg_run

    create_env, env_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(env_name, create_env)

    test_env = create_env()
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def gen_policy():
        return PPOTFPolicy, obs_space, act_space, {}

    # Setup PG with a single policy graph for all agents
    policy_graphs = {"av": gen_policy()}

    def policy_mapping_fn(_):
        return "av"

    config.update(
        {
            "multiagent": {
                "policies": policy_graphs,
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["av"],
            }
        }
    )

    return alg_run, env_name, config


if __name__ == "__main__":
    EXAMPLE_USAGE = """
    example usage:
        python multiagent_traffic_light_grid.py --upload_dir=<S3 bucket>
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="[Flow] Issues multi-agent traffic light grid experiment",
        epilog=EXAMPLE_USAGE,
    )

    # required input parameters
    parser.add_argument(
        "--upload_dir", type=str, help="S3 Bucket for uploading results."
    )

    # optional input parameters
    parser.add_argument(
        "--run_mode",
        type=str,
        default="local",
        help="Experiment run mode (local | cluster)",
    )
    parser.add_argument(
        "--algo", type=str, default="PPO", help="RL method to use (PPO)"
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        default=3,
        help="The number of rows in the traffic light grid network.",
    )
    parser.add_argument(
        "--num_cols",
        type=int,
        default=3,
        help="The number of columns in the traffic light grid network.",
    )
    parser.add_argument(
        "--inflow_rate",
        type=int,
        default=300,
        help="The inflow rate (veh/hr) per edge.",
    )
    parser.add_argument('--env',
            default='BasicEnv',
            choices=['BasicEnv', 'RewardSharingEnv', 'RewardSharingEnvSimple', 'RewardSharingEnvColight'],
            help='The environment to use to run the simulation')
    parser.add_argument('--k_nearest',
            type=int,
            default=5,
            help='The number of nearest neighbors to be used for colight')
    parser.add_argument('--temp',
            type=float,
            default=0.5,
            help='The temperature to be used for colight')
    parser.add_argument('--neighbor_weight',
            type=float,
            default=0.1,
            help='The neighbor weight used for reward sharing')
    parser.add_argument('--password',
            default='password.txt',
            help='Password file to be used for redis')
    args = parser.parse_args()
    envs = {'BasicEnv':BasicEnv,
            'RewardSharingEnv': RewardSharingEnv,
            'RewardSharingEnvSimple':RewardSharingEnvSimple,
            'RewardSharingEnvKNN':RewardSharingEnvKNN
            }

    EDGE_INFLOW = args.inflow_rate  # inflow rate of vehicles at every edge
    N_ROWS = args.num_rows  # number of row of bidirectional lanes
    N_COLUMNS = args.num_cols  # number of columns of bidirectional lanes

    flow_params = make_flow_params(N_ROWS, N_COLUMNS, EDGE_INFLOW, exp_env=envs[args.env])

    upload_dir = args.upload_dir
    RUN_MODE = args.run_mode
    ALGO = args.algo

    if ALGO == "PPO":
        alg_run, env_name, config = setup_exps_PPO(flow_params)
    else:
        raise NotImplementedError

    with open('password.txt' ,'r') as f:
        password = f.readline()

    if RUN_MODE == "local":
        ray.init(num_cpus=N_CPUS + 1, redis_password=password)
        N_ITER = 1000
    elif RUN_MODE == "cluster":
        ray.init(redis_address="localhost:6379")
        N_ITER = 2000

    exp_tag = {
        "run": alg_run,
        "env": env_name,
        "checkpoint_freq": 25,
        "max_failures": 10,
        "stop": {"training_iteration": N_ITER},
        "config": config,
        "num_samples": 1,
    }

    if upload_dir:
        exp_tag["upload_dir"] = "s3://{}".format(upload_dir)

    run_experiments({flow_params["exp_tag"]: exp_tag},)
