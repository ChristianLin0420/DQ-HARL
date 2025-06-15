import time
import numpy as np
from harl.common.base_logger import BaseLogger


class GYMLogger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, wandb_run, run_dir):
        super(GYMLogger, self).__init__(
            args, algo_args, env_args, num_agents, writter, wandb_run, run_dir
        )

    def get_task_name(self):
        return self.env_args["scenario"]
