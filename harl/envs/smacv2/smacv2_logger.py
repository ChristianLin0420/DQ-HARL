import time
from functools import reduce
import numpy as np
from harl.common.base_logger import BaseLogger
from harl.envs.smac.smac_logger import SMACLogger


class SMACv2Logger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, wandb_run, run_dir):
        super(SMACv2Logger, self).__init__(
            args, algo_args, env_args, num_agents, writter, wandb_run, run_dir
        )
        self.win_key = "battle_won"
