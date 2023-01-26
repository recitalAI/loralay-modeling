import torch 

from transformers import Seq2SeqTrainer 
from transformers.optimization import (
    get_scheduler,
    get_polynomial_decay_schedule_with_warmup
)
from transformers.trainer_utils import SchedulerType


class LoRaLayTrainer(Seq2SeqTrainer):
    def create_scheduler(
        self, 
        num_training_steps: int, 
        optimizer: torch.optim.Optimizer = None, 
    ):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.
        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            # Polynomial decay of the learning rate
            if self.args.power != 1.0 and self.args.lr_scheduler_type == SchedulerType("polynomial"):
                self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    lr_end=self.args.lr_end,
                    power=self.args.power,
                )
            else:
                self.lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
        return self.lr_scheduler
