"""Configuration assembly and experiment bookkeeping."""

import subprocess
from pathlib import Path
import logging
import os
import sys
import shutil
import yaml
import socket
from shlex import quote
from datetime import datetime
from glob import glob

import habitat
import numpy as np
import torch
import clip
import PIL
import lmdb
from torchvision import transforms
from habitat_baselines.config.default import get_config as get_habitat_config

logger = logging.getLogger(__name__)


from habitat.tasks.nav.instance_image_nav_task import (
    InstanceImageGoalSensor
)
from src.task_sensors import (
    CachedGoalFeatureSensor,
    ReferenceObjectSensor,
    TargetViewSensor,
)
from habitat.config import Config


def find_config_differences(current_config, previous_config):
    def flatten_tree(src, target=None, prefix=""):
        if target is None:
            target = {}
        for k, v in src.items():
            if type(v) is dict:
                flatten_tree(v, target, prefix + k + ".")
            else:
                target[prefix+k] = v
        return target

    # load previous config and find unmatch config
    current_config = yaml.load(str(current_config), Loader=yaml.FullLoader)
    previous_config = yaml.load(str(previous_config), Loader=yaml.FullLoader)
    unmatch_config = {
        key: value
        for key, value in flatten_tree(previous_config).items()
        if flatten_tree(current_config)[key] != value
    }
    return unmatch_config


def write_replay_script(run_dir, run_type):
    with open(run_dir / 'code/run_{}_{}.sh'.format(run_type, socket.gethostname()), 'w') as f:
        f.write(f'cd {quote(os.getcwd())}\n')
        f.write('mkdir -p {}\n'.format(run_dir / 'code/unpack'))
        f.write('tar -C {} -xzvf {}\n'.format(run_dir / 'code/unpack', run_dir / 'code/code_*.tar.gz'))
        f.write('cd {}\n'.format(run_dir / 'code/unpack'))
        f.write('patch -p1 < ../dirty.patch\n')
        f.write(f'cd {quote(os.getcwd())}\n')
        f.write('cp -r -f {} {}\n'.format(run_dir / 'code/unpack/*', quote(os.getcwd())))
        envs = ['CUDA_VISIBLE_DEVICES']
        for env in envs:
            value = os.environ.get(env, None)
            if value is not None:
                f.write(f'export {env}={quote(value)}\n')
        f.write(sys.executable + ' ' + ' '.join(quote(arg) for arg in sys.argv) + '\n')


def write_config_snapshot(run_dir, config, snapshot_kind):
    if config is not None:
        output_file = open(run_dir / 'config_of_{}.yaml'.format(snapshot_kind), 'w')
        output_file.write(str(config))
        output_file.close()


def allocate_run_directory(exp_dir: Path, prefix: str = 'run', suffix: str = ''):
    if exp_dir.exists():
        runs = glob(str(exp_dir / '*_run*'))
        num_runs = len([r for r in runs if (prefix + '_') in r])
    else:
        num_runs = 0
    dt = datetime.now().strftime('%Y%m%d-%H%M%S')
    rundir = prefix + '_' + 'run{}'.format(num_runs) + '_' + suffix
    return (exp_dir / rundir)


def archive_source_tree(run_dir: str):
    run_dir = Path(run_dir) / 'code'
    if not run_dir.exists():
        run_dir.mkdir()
    if os.path.isdir(".git"):
        HEAD_commit_id = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            check=True, stdout=subprocess.PIPE, text=True
        )
        tar_name = f'code_{HEAD_commit_id.stdout[:-1]}.tar.gz'
        subprocess.run(
            ['git', 'archive', '-o', str(run_dir/tar_name), 'HEAD'],
            check=True,
        )
        diff_process = subprocess.run(
            ['git', 'diff', 'HEAD'],
            check=True, stdout=subprocess.PIPE, text=True,
        )
        if diff_process.stdout:
            logger.warning('Working tree is dirty. Patch:\n%s', diff_process.stdout)
            with (run_dir / 'dirty.patch').open('w') as f:
                f.write(diff_process.stdout)
    else:
        logger.warning('.git does not exist in current dir')


def attach_pose_measures(config):
    with habitat.config.read_write(config):
        if TargetViewSensor.cls_uuid in config.habitat.task.sensors:
            goalsensoruuid = TargetViewSensor.cls_uuid
        elif ReferenceObjectSensor.cls_uuid in config.habitat.task.sensors:
            goalsensoruuid = ReferenceObjectSensor.cls_uuid
        elif InstanceImageGoalSensor.cls_uuid + "_sensor" in config.habitat.task.sensors:
            goalsensoruuid = InstanceImageGoalSensor.cls_uuid
        else:
            assert False,"Do not specifit goal sensor"

        config.habitat.task.pose_goal_distance = Config()
        config.habitat.task.pose_goal_distance.type = "PoseGoalDistance"
        config.habitat.task.pose_goal_distance.goalsensoruuid = goalsensoruuid

        config.habitat.task.pose_goal_success = Config()
        config.habitat.task.pose_goal_success.type = "PoseGoalSuccess"
        config.habitat.task.pose_goal_success.view_weight = 0.5
        config.habitat.task.pose_goal_success.angle_threshold = 25.0
        config.habitat.task.pose_goal_success.goalsensoruuid = goalsensoruuid

        config.habitat.task.terminal_view_angle = Config()
        config.habitat.task.terminal_view_angle.type = "TerminalViewAngle"
        config.habitat.task.terminal_view_angle.goalsensoruuid = goalsensoruuid
    return config


def assemble_experiment_config(
    exp_config, opts, run_type, model_dir, overwrite, note, debug, global_rank
):
    exp_config = exp_config.split(',')
    exp_config = [path if '.yaml' in path else f'src/config/{path}.yaml' for path in exp_config]

    config = get_habitat_config(exp_config, opts)
    config = attach_pose_measures(config)
    config.defrost()

    if model_dir == None:
        model_dir = 'results/official'

    config.habitat_baselines.checkpoint_folder = os.path.join(model_dir, 'ckpts')
    if 'habitat_baselines.eval_ckpt_path_dir' not in opts:
        config.habitat_baselines.eval_ckpt_path_dir = os.path.join(model_dir, 'ckpts')
    config.habitat_baselines.tensorboard_dir = os.path.join(model_dir, 'tb')
    config.habitat_baselines.video_dir = os.path.join(model_dir, "video")

    if debug:
        ds_type = config.habitat.environment.type
        if ds_type == "gibson":
            scene = getattr(config.habitat, 'debug_scenes', ['Adrian'])
        elif ds_type == "mp3d":
            scene = getattr(config.habitat, 'debug_scenes', ['pRbA3pwrgk9'])
        elif ds_type == "hm3d":
            scene = getattr(config.habitat, 'debug_scenes', ['00001-UVdNNRcVyV1'])
        else:
            scene = getattr(config.habitat, 'debug_scenes', ['NRsmXFcVTbN'])

        logger.warning('Debug using 1 scene!')
        config.habitat.dataset.content_scenes = scene
        config.habitat_baselines.log_interval = 1


    if global_rank == 0:
        if overwrite:
            logger.warning('Warning! overwrite is specified!\nCurrent model dir will be removed!')
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir, ignore_errors=True)
                os.makedirs(model_dir, exist_ok=True)

        run_dir = allocate_run_directory(
            exp_dir=Path(model_dir), prefix=run_type, suffix=note
        )
        
        config.habitat_baselines.log_file = os.path.join(run_dir, 'log.txt')
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(config.habitat_baselines.tensorboard_dir, exist_ok=True)
        os.makedirs(config.habitat_baselines.checkpoint_folder, exist_ok=True)

        archive_source_tree(run_dir)
        write_replay_script(run_dir, run_type)
        write_config_snapshot(run_dir, config, run_type)

    config.freeze()
    return config
