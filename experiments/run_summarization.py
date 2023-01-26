import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock
from transformers import (
    PegasusConfig,
    BigBirdPegasusConfig, 
    MBartConfig,
    T5Config,
    AutoTokenizer,
    MBartTokenizer,
    MBartTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    PegasusForConditionalGeneration,
    BigBirdPegasusForConditionalGeneration,
    MBartForConditionalGeneration,
    T5ForConditionalGeneration,
    LEDConfig,
    LEDForConditionalGeneration,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from loralay.training.trainers import LoRaLayTrainer
from loralay.training.training_args_loralay import LoRaLaySummarizationTrainingArguments

from loralay.modeling.layout_pegasus import (
    LayoutPegasusConfig,
    LayoutPegasusForConditionalGeneration
)
from loralay.modeling.layout_mbart import (
    LayoutMBartConfig,
    LayoutMBartForConditionalGeneration
)
from loralay.modeling.layout_bigbird_pegasus import (
    LayoutBigBirdPegasusConfig,
    LayoutBigBirdPegasusForConditionalGeneration
)
from loralay.modeling.layout_led import (
    LayoutLEDConfig,
    LayoutLEDForConditionalGeneration
)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


MODEL_CLASSES = {
    "pegasus": (PegasusConfig, PegasusForConditionalGeneration),
    "layout_pegasus": (LayoutPegasusConfig, LayoutPegasusForConditionalGeneration),
    "bigbird_pegasus": (BigBirdPegasusConfig, BigBirdPegasusForConditionalGeneration),
    "layout_bigbird_pegasus": (LayoutBigBirdPegasusConfig, LayoutBigBirdPegasusForConditionalGeneration),
    "mbart": (MBartConfig, MBartForConditionalGeneration),
    "layout_mbart": (LayoutMBartConfig, LayoutMBartForConditionalGeneration),
    "bigbird_mbart": (BigBirdPegasusConfig, BigBirdPegasusForConditionalGeneration),
    "layout_bigbird_mbart": (LayoutBigBirdPegasusConfig, LayoutBigBirdPegasusForConditionalGeneration),
    "t5": (T5Config, T5ForConditionalGeneration),
    "led": (LEDConfig, LEDForConditionalGeneration),
    "layout_led": (LayoutLEDConfig, LayoutLEDForConditionalGeneration),
}

DATASET2HFNAME = {
    "arxiv_lay": "arxiv-summarization",
    "pubmed_lay": "pubmedlay-summarization",
    "hal": "hal-summarization",
    "scielo_es": "scielo-summarization",
    "scielo_pt": "scielo-summarization",
    "koreascience": "koreascience-summarization",
}

DATASET_TO_LID = {
    "arxiv_lay": "en_XX",
    "pubmed_lay": "en_XX",
    "hal": "fr_XX",
    "scielo_es": "es_XX",
    "scielo_pt": "pt_XX",
    "koreascience": "ko_KR",
}

_PADDING_BBOX = [0, 0, 0, 0]

MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_type: str = field(
        metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys())}
    )
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    model_name_or_path_for_layout: Optional[str] = field(
        default=None, metadata={"help": "If `model_type` = layout-bigbird-pegasus, path to pretrained model or model identifier from huggingface.co/models, used to initialize the layout embeddings."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use, selected in the list: " + ", ".join(DATASET2HFNAME.keys())}
    )
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "The path to the data directory."}
    )
    cached_data_dir: str = field(
        default=None,
        metadata={"help": "The path to the cached features"}
    )
    download_mode: Optional[str] = field(
        default="reuse_dataset_if_exists",
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    start_idx_predict_samples: Optional[int] = field(
        default=0,
        metadata={
            "help": "For debugging purposes or quicker training, index specifying at which position to start "
            "selecting prediction examples."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
            "Useful for multilingual models like mBART where the first generated token"
            "needs to be the target language token (Usually it is the target language token)"
        },
    )

    path_to_metric: Optional[str] = field(
        default=None, metadata={"help": "Path to metric file."}
    )
    train_processed_cache_file_name: Optional[str] = field(
        default=None, 
        metadata={
            "help": "The name of a path for the cache file storing the processed train dataset."
        }
    )
    val_processed_cache_file_name: Optional[str] = field(
        default=None, 
        metadata={
            "help": "The name of a path for the cache file storing the processed validation dataset."
        }
    )
    test_processed_cache_file_name: Optional[str] = field(
        default=None, 
        metadata={
            "help": "The name of a path for the cache file storing the processed test dataset."
        }
    )

    length_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "Exponential penalty to the length. 1.0 means no penalty. "
                    "Set to values < 1.0 in order to encourage the model to generate shorter sequences, "
                    "to a value > 1.0 in order to encourage the model to produce longer sequences."
        }
    )

    def __post_init__(self):
        if self.data_dir is None:
            raise ValueError("Need a data path.")
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LoRaLaySummarizationTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_args.dataset_name = data_args.dataset_name.lower()
    assert (
        data_args.dataset_name in DATASET2HFNAME
    ), f'`dataset_name` must be selected in the list {", ".join(DATASET2HFNAME.keys())}'


    if data_args.dataset_name not in ["scielo_es", "scielo_pt"]:
        raw_datasets = load_dataset(
            f"nglaura/{DATASET2HFNAME[data_args.dataset_name])}",
        )
    else:
        raw_datasets = load_dataset(
            f"nglaura/{DATASET2HFNAME[data_args.dataset_name])}",
            data_args.dataset_name
        )


    model_args.model_type = model_args.model_type.lower()
    assert (
        model_args.model_type in MODEL_CLASSES
    ), f'`model_type` must be selected in the list {", ".join(MODEL_CLASSES.keys())}'

    config_class, model_class = MODEL_CLASSES[model_args.model_type]

    config = config_class.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model_args.model_type in [
        "bigbird_mbart",
        "layout_bigbird_mbart",
        "led",
        "layout_led"
    ]:
        config.max_position_embeddings = 4096 
        config.max_length = data_args.max_target_length
    elif model_args.model_type == "mbart":
        config.max_length = data_args.max_target_length
    elif model_args.model_type == "t5":
        config.n_positions = 1024

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        add_prefix_space=True if model_args.model_type in ["led", "layout_led"] else None,
    )

    if "layout" in model_args.model_type:
        tokenizer.model_input_names = tokenizer.model_input_names + ["bbox"]

    def load_weights_into_bigbird():
        bigbird_model = model_class(config=config)

        with torch.no_grad():
            # Token embeddings
            bigbird_model.model.shared.load_state_dict(
                source_model.model.shared.state_dict()
            )
            bigbird_model.model.encoder.embed_tokens.load_state_dict(
                source_model.model.encoder.embed_tokens.state_dict()
            ) 

            if "pegasus" in model_args.model_type:
                encoder_embed_positions_to_copy = source_model.model.encoder.embed_positions.weight
                decoder_embed_positions_to_copy = source_model.model.decoder.embed_positions.weight
            else:
                encoder_embed_positions_to_copy = source_model.model.encoder.embed_positions.weight[2:]
                decoder_embed_positions_to_copy = source_model.model.decoder.embed_positions.weight[2:]

            # Position embeddings
            bigbird_model.model.encoder.embed_positions.weight[:source_model.config.max_position_embeddings, :].copy_(
                encoder_embed_positions_to_copy
            ) 

            # Layer Normalization
            bigbird_model.model.encoder.layernorm_embedding.load_state_dict(
                source_model.model.encoder.layer_norm.state_dict()
            )

            # Encoder layers
            for i in range(len(bigbird_model.model.encoder.layers)):
                # Self-attention weights
                self_attention_mapping = {
                    bigbird_model.model.encoder.layers[i].self_attn.self.query: source_model.model.encoder.layers[i].self_attn.q_proj,
                    bigbird_model.model.encoder.layers[i].self_attn.self.key: source_model.model.encoder.layers[i].self_attn.k_proj,
                    bigbird_model.model.encoder.layers[i].self_attn.self.value: source_model.model.encoder.layers[i].self_attn.v_proj
                }
                for target, origin in self_attention_mapping.items():
                    target.load_state_dict(origin.state_dict(), strict=False) # bias is set to False in BigBirdPegasusEncoderAttention.self_attn
                bigbird_model.model.encoder.layers[i].self_attn.output.load_state_dict(
                    source_model.model.encoder.layers[i].self_attn.out_proj.state_dict(), strict=False 
                ) # bias is set to False in BigBirdPegasusEncoderAttention.self_attn
                
                # Layer normalization 
                bigbird_model.model.encoder.layers[i].self_attn_layer_norm.load_state_dict(
                    source_model.model.encoder.layers[i].self_attn_layer_norm.state_dict()
                )
                # Linear layers
                bigbird_model.model.encoder.layers[i].fc1.load_state_dict(
                    source_model.model.encoder.layers[i].fc1.state_dict()
                )
                bigbird_model.model.encoder.layers[i].fc2.load_state_dict(
                    source_model.model.encoder.layers[i].fc2.state_dict()
                )
                # Layer normalization 
                bigbird_model.model.encoder.layers[i].final_layer_norm.load_state_dict(
                    source_model.model.encoder.layers[i].final_layer_norm.state_dict()
                )

            # Token embeddings
            bigbird_model.model.decoder.embed_tokens.load_state_dict(
                source_model.model.decoder.embed_tokens.state_dict(), 
            )   
            # Position embeddings
            bigbird_model.model.decoder.embed_positions.weight[:source_model.config.max_position_embeddings, :].copy_(
                decoder_embed_positions_to_copy
            )
            # Layer normalization
            bigbird_model.model.decoder.layernorm_embedding.load_state_dict(
                source_model.model.decoder.layer_norm.state_dict()
            )

            # Copy whole decoder layers (BigBirdPegasusDecoder's structure is the same as PegasusDecoder's)
            for i in range(len(bigbird_model.model.decoder.layers)):
                bigbird_model.model.decoder.layers[i].load_state_dict(
                    source_model.model.decoder.layers[i].state_dict(), strict=False
                ) # Bias is set to False in BigBirdPegasusDecoderAttention.self_attn

            bigbird_model.lm_head.load_state_dict(
                source_model.lm_head.state_dict()
            )
        
        return bigbird_model


    if (
        model_args.model_type in [
            "bigbird_pegasus", 
            "layout_bigbird_pegasus",
            "bigbird_mbart",
            "layout_bigbird_mbart",
        ]
        and training_args.do_train
    ): # BigBird model
        if "pegasus" in model_args.model_type:
            source_model = PegasusForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            source_model = MBartForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        
        model = load_weights_into_bigbird()
        
        del source_model 
        torch.cuda.empty_cache()
    else:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    
    model.resize_token_embeddings(len(tokenizer))
    
    model.config.early_stopping = True
    if data_args.length_penalty is not None:
        model.config.length_penalty = data_args.length_penalty

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[DATASET_TO_LID[data_args.dataset_name]]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(DATASET_TO_LID[data_args.dataset_name])
            

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )


    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(model.config)

    if training_args.gradient_checkpointing:
        model.config.use_cache = False
        
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        tokenizer.src_lang = DATASET_TO_LID[data_args.dataset_name]
        tokenizer.tgt_lang = DATASET_TO_LID[data_args.dataset_name]

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    text_column = "article_words"
    summary_column = "abstract"
    bbox_column = "article_norm_bboxes"

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        # remove pairs where at least one record is None
        inputs, input_bboxes, targets = [], [], []
        for i in range(len(examples[text_column])):
            if (
                examples[text_column][i] is not None 
                and examples[bbox_column][i] is not None
                and examples[summary_column][i] is not None
            ):
                inputs.append(examples[text_column][i])
                input_bboxes.append(examples[bbox_column][i])
                targets.append(examples[summary_column][i])

        if len(prefix) > 0:
            inputs = [[prefix] + inp for inp in inputs]
            input_bboxes = [_PADDING_BBOX + inp_bbox for inp_bbox in input_bboxes]
        
        model_inputs = tokenizer(
            inputs,
            padding=padding,
            max_length=data_args.max_source_length,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)


        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]

        bboxes = []
        
        for batch_index in range(len(model_inputs["input_ids"])):
            word_ids = model_inputs.word_ids(batch_index=batch_index)
            bbox = examples[bbox_column][batch_index]
            bbox_inputs = []

            for word_idx in word_ids:
                if word_idx is None:
                    bbox_inputs.append(_PADDING_BBOX)
                else:
                    bbox_inputs.append(bbox[word_idx])
            bboxes.append(bbox_inputs)

        model_inputs["bbox"] = bboxes

        return model_inputs


    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                cache_file_name=data_args.train_processed_cache_file_name,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                cache_file_name=data_args.val_processed_cache_file_name,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
            if data_args.start_idx_predict_samples + data_args.max_predict_samples < len(predict_dataset):
                end_idx_predict_samples = data_args.start_idx_predict_samples + data_args.max_predict_samples
            else:
                end_idx_predict_samples = len(predict_dataset)
            predict_dataset = predict_dataset.select(
                range(
                    data_args.start_idx_predict_samples, 
                    end_idx_predict_samples
                )
            )
            print(len(predict_dataset))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                cache_file_name=data_args.test_processed_cache_file_name,
                desc="Running tokenizer on prediction dataset",
            )
        print(len(predict_dataset))


    if data_args.dataset_name == "hal":
        if model_args.model_type in ["mbart", "layout-mbart"]:
            model.config.num_beams = 8
        elif model_args.model_type in ["bigbird-mbart", "layout-bigbird-mbart"]:
            model.config.num_beams = 5
        model.config.length_penalty = 0.8

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        max_length=data_args.max_source_length,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        label_pad_token_id=label_pad_token_id,
    )

    # Metric
    metric_name_or_path = "rouge" if not data_args.path_to_metric else data_args.path_to_metric
    metric = load_metric(metric_name_or_path)

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    trainer = LoRaLayTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))


        return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
