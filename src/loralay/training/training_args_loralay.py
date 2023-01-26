import logging
from dataclasses import dataclass, field
from typing import Optional

from transformers import Seq2SeqTrainingArguments

logger = logging.getLogger(__name__)

@dataclass
class LoRaLaySummarizationTrainingArguments(Seq2SeqTrainingArguments):
    """
    lr_end (`float`, *optional*, defaults to 1e-7):
        The end LR for polynomial learning rate decay..
    power (`float`, *optional*, defaults to 1.0):
        The power factor for polynomial learning rate decay.
    """

    lr_end: Optional[float] = field(
        default=1e-7, 
        metadata={"help": "The end LR for polynomial learning rate decay."}
    )
    power: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "The power factor for polynomial learning rate decay."
        },
    )