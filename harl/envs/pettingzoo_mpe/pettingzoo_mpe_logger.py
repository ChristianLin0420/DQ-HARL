import time
import numpy as np
from harl.common.base_logger import BaseLogger


class PettingZooMPELogger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, wandb_run, run_dir):
        super(PettingZooMPELogger, self).__init__(
            args, algo_args, env_args, num_agents, writter, wandb_run, run_dir
        )

    def get_task_name(self):
        if self.env_args["continuous_actions"]:
            return f"{self.env_args['scenario']}-continuous"
        else:
            return f"{self.env_args['scenario']}-discrete"
