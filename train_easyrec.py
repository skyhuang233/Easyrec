import os
import torch
import random
import logging
import numpy as np
import transformers

seed=2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from model import Easyrec
from utility.logger import *
from utility.metric import *
from utility.trainer import *
from datetime import datetime
from utility.load_data import *
from transformers import AutoConfig
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


from dataclasses import dataclass, field
from typing import Optional
import transformers


# -----------------------------
# 模型参数
# -----------------------------
@dataclass
class ModelArguments:
    """Model configuration and training strategy"""
    model_name_or_path: Optional[str] = field(
        default="roberta-base",
        metadata={"help": "Pretrained model name or path"}
    )
    model_size: str = field(
        default="base",
        metadata={"help": "Model size, e.g., base or large"}
    )
    loss: str = field(
        default="contrastive_mlm",
        metadata={"help": "Loss type, e.g., contrastive, mlm, contrastive_mlm"}
    )
    temperature: float = field(
        default=0.07,
        metadata={"help": "Temperature for contrastive loss"}
    )
    mlm_weight: float = field(
        default=0.2,
        metadata={"help": "Weight for MLM objective when used together with contrastive loss"}
    )
    pooler_type: str = field(
        default="cls",
        metadata={"help": "Pooling strategy (cls, avg, etc.)"}
    )
    do_mlm: bool = field(
        default=False,
        metadata={"help": "Enable MLM during training"}
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={"help": "Use MLP projection only during training"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache directory for pretrained models"}
    )
    use_auth_token: bool = field(default=False)
    model_revision: str = field(default="main")
    use_fast_tokenizer: bool = field(default=True)


# -----------------------------
# 数据参数
# -----------------------------
@dataclass
class DataArguments:
    """Dataset configuration"""
    train_data: str = field(
        default=None,
        metadata={"help": "Path to training data CSV file"}
    )
    val_data: str = field(
        default=None,
        metadata={"help": "Path to validation data CSV file"}
    )
    user_profiles: str = field(
        default=None,
        metadata={"help": "Path to user profiles JSONL file"}
    )
    item_profiles: str = field(
        default=None,
        metadata={"help": "Path to item profiles JSONL file"}
    )
    max_history: int = field(
        default=20,
        metadata={"help": "Maximum length of user interaction history"}
    )
    add_item_raw_meta: bool = field(
        default=True,
        metadata={"help": "Whether to include raw item metadata"}
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "MLM masking probability (if MLM is used)"}
    )
    data_path: str = field(default="data/")
    used_diverse_profile_num: int = field(default=3)
    total_diverse_profile_num: int = field(default=3)


# -----------------------------
# 训练参数
# -----------------------------
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Training configuration"""
    batch_size: int = field(
        default=64,
        metadata={"help": "Batch size per device"}
    )
    grad_accum: int = field(
        default=2,
        metadata={"help": "Gradient accumulation steps"}
    )
    epochs: int = field(
        default=10,
        metadata={"help": "Total number of training epochs"}
    )
    learning_rate: float = field(
        default=3e-5,
        metadata={"help": "Initial learning rate"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay for optimizer"}
    )
    amp: str = field(
        default="bf16",
        metadata={"help": "Automatic mixed precision mode (fp16, bf16, or none)"}
    )
    num_workers: int = field(
        default=8,
        metadata={"help": "Number of data loader workers"}
    )
    output_dir: str = field(
        default="checkpoints/easyrec_base",
        metadata={"help": "Output directory for model checkpoints"}
    )
    log_dir: str = field(
        default="logs/easyrec_base",
        metadata={"help": "Directory to save training logs"}
    )
    seed: int = field(
        default=2025,
        metadata={"help": "Random seed for reproducibility"}
    )


def main():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ---------- 映射自定义参数到 HF 实际使用字段（务必放在解析后、logger 之前） ----------
    # batch / epochs / grad_accum / workers / log_dir
    training_args.per_device_train_batch_size = getattr(training_args, "batch_size", training_args.per_device_train_batch_size)
    training_args.per_device_eval_batch_size  = training_args.per_device_train_batch_size
    training_args.gradient_accumulation_steps = getattr(training_args, "grad_accum", training_args.gradient_accumulation_steps)
    training_args.num_train_epochs            = float(getattr(training_args, "epochs", training_args.num_train_epochs))
    training_args.dataloader_num_workers      = getattr(training_args, "num_workers", training_args.dataloader_num_workers)
    training_args.logging_dir                 = getattr(training_args, "log_dir", training_args.logging_dir)

    # AMP 开关
    amp = str(getattr(training_args, "amp", "")).lower()
    training_args.bf16 = (amp == "bf16") or training_args.bf16
    training_args.fp16 = (amp == "fp16") or training_args.fp16

    # 根据 AMP 选择 torch_dtype
    torch_dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)

    # 离线模式（已 export TRANSFORMERS_OFFLINE=1 则生效）
    offline = os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"

    local_rank = training_args.local_rank
    print("training_args.output_dir", training_args.output_dir)

    # ---------- logger（放在映射之后） ----------
    logger = EasyrecEmbedderTrainingLogger(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    logger.log(model_args)
    logger.log(data_args)
    logger.log(training_args)

    # ---------- load model ----------
    if 'roberta' in model_args.model_name_or_path:
        config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
            "local_files_only": offline,  # 显式离线
        }
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
        model = Easyrec.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            model_args=model_args,
            torch_dtype=torch_dtype,
            local_files_only=offline,      # 显式离线（若继承 HF 模型则会生效）
        )
    else:
        raise NotImplementedError

    # ---------- tokenizer ----------
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        padding_side="right",
        use_fast=False,               # 如果想用 fast，可改为 model_args.use_fast_tokenizer
        local_files_only=offline,     # 显式离线
    )

    # ---------- data module ----------
    data_module = make_pretrain_embedder_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # ---------- trainer ----------
    trainer = EasyrecEmbedderTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )
    metric = Metric(metrics=['recall'], k=[20])
    trainer.add_evaluator(metric)
    trainer.add_logger(logger)

    # ---------- training ----------
    trainer.train()


if __name__ == "__main__":
    main()