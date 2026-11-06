from ..base import BaseConfig


class TaskConfig(BaseConfig):
    def __init__(
        self,
        major_task: str,
        minor_task: str,
        version: str,
        task_description: str,
        device: str,
        experiments_dir: str,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
    ):
        super().__init__()
        self.capture_config()
