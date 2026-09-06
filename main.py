#!/usr/bin/env python3

"""Command-line entry point for navigation experiments."""

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import argparse
import random
from typing import Optional

import numpy as np
import torch
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:100'
os.environ['HABITAT_SIM_LOG'] = 'quiet'
os.environ['MAGNUM_LOG'] = 'quiet'

from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.ddp_utils import get_distrib_size

from src import experiment_setup
import src.agent_policy as _agent_policy  # noqa: F401 - registry side effect
import src.distributed_trainer as _distributed_trainer  # noqa: F401
import src.episode_data as _episode_data  # noqa: F401
import src.navigation_measures as _navigation_measures  # noqa: F401
import src.task_environment as _task_environment  # noqa: F401
import src.task_sensors as _task_sensors  # noqa: F401

_REGISTERED_COMPONENTS = (
    _agent_policy,
    _distributed_trainer,
    _episode_data,
    _navigation_measures,
    _task_environment,
    _task_sensors,
)


def create_argument_parser(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    parser.add_argument(
        "--model-dir",
        default="results/imagenav/early-fusion-r9",
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "traverse"],
        # required=True,
        default="eval",
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        # required=True,
        default=(
            "exp_config/gibson_experiment.yaml,agent,task_objective,"
            "gibson_source,task_observations,context_prior,evaluation"
        ),
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action='store_true',
        help="Modify config options from command line"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action='store_true',
        help="debug using 1 scene"
    )
    parser.add_argument(
        "--note",
        default="",
        help="Add extra note for running file"
    )

    # distributed training parameters
    # parser.add_argument('--local_rank', type=int, default=-1)

    return parser


def dispatch_experiment(config: Config, run_type: str) -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """  


    random.seed(config.habitat.seed)
    np.random.seed(config.habitat.seed)
    torch.manual_seed(config.habitat.seed)
    if (
        config.habitat_baselines.force_torch_single_threaded
        and torch.cuda.is_available()
    ):
        torch.set_num_threads(1)

    trainer_factory = baseline_registry.get_trainer(
        config.habitat_baselines.trainer_name
    )
    assert (
        trainer_factory is not None
    ), f"{config.habitat_baselines.trainer_name} is not supported" 
    experiment_runner = trainer_factory(config)

    # if not os.path.exists(os.path.dirname(config.habitat_baselines.log_file)):
    #     os.makedirs(os.path.dirname(config.habitat_baselines.log_file))
    #     f=open(config.habitat_baselines.log_file,"w")
    #     f.close()


    if run_type == "train":
        experiment_runner.train()
    elif run_type == "eval":
        experiment_runner.eval()
    elif run_type == "traverse":
        experiment_runner.traverse()


def launch_experiment(
    exp_config: str,
    run_type: str,
    opts=None,
    model_dir=None,
    overwrite=False,
    note=None,
    debug=False,
    local_rank=None,
) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval".
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    # print(opts)

    local_rank, world_rank, _ = get_distrib_size()
    config = experiment_setup.assemble_experiment_config(
        exp_config,
        opts,
        run_type,
        model_dir,
        overwrite,
        note,
        debug,
        world_rank,
    )
    device = torch.device(f'cuda:{local_rank}')
    print(device)
    torch.cuda.set_device(device)

    dispatch_experiment(config, run_type)


def main() -> None:
    arguments = create_argument_parser().parse_args()
    launch_experiment(**vars(arguments))


if __name__ == "__main__":
    main()
