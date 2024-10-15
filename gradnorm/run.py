import logging
import os
import json
from pathlib import Path
import torch.distributed as dist
import sys 
sys.path.append(os.path.abspath("."))

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl
)

from arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
from data import SameDatasetTrainDataset, EmbedCollator 
from modeling import BGEM3Model
from trainer import BiTrainer


class TrainerCallbackForDataRefresh(TrainerCallback):
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset
        
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        self.train_dataset.refresh_epoch()

    
class TrainerCallbackForLog(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger
        
    def on_pre_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_log = True
        
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.logger.info(state.log_history[-1])
        
        
def get_logger(logging_pth):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    parent = Path(logging_pth).parent
    if not parent.exists():
        os.makedirs(parent, exist_ok=True)
    file_handler = logging.FileHandler(logging_pth)    
    logger.addHandler(file_handler)
    return logger 


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    
    # Setup logger
    logger = get_logger(data_args.logging_pth)

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model = BGEM3Model(model_name=model_args.model_name_or_path,
                       normlized=training_args.normlized,
                       sentence_pooling_method=training_args.sentence_pooling_method,
                       negatives_cross_device=training_args.negatives_cross_device,
                       temperature=training_args.temperature,
                       enable_sub_batch=training_args.enable_sub_batch,
                       unified_finetuning=training_args.unified_finetuning,
                       use_self_distill=training_args.use_self_distill,
                       colbert_dim=training_args.colbert_dim,
                       self_distill_start_step=training_args.self_distill_start_step,
                       logger=logger)

    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False
    if training_args.fix_encoder:
        for k, v in model.named_parameters():
            if "colbert_linear" in k or 'sparse_linear' in k:
                logging.info(f"train the parameters for {k}")
            else:
                v.requires_grad = False

    if data_args.same_task_within_batch:
        training_args.per_device_train_batch_size = 1
        train_dataset = SameDatasetTrainDataset(args=data_args, 
                                                batch_size=training_args.per_device_train_batch_size, 
                                                seed=training_args.seed, 
                                                num_processes=training_args.world_size,
                                                process_index=training_args.process_index,
                                                logger=logger,
                                                )
        training_args.dataloader_num_workers = 0    # avoid multi-processes
    else:
        raise NotImplementedError("Not support `same_task_within_batch=False`")

    data_collator = EmbedCollator(
        tokenizer,
        query_max_len=data_args.query_max_len,
        passage_max_len=data_args.passage_max_len
    )
    
    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    if data_args.same_task_within_batch:
        trainer.add_callback(TrainerCallbackForDataRefresh(train_dataset))
    trainer.add_callback(TrainerCallbackForLog(logger))
    
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Dataset: %s", data_args.train_data)
    logger.info("Model: %s", model_args.model_name_or_path)
    logger.info("temperature: %s", str(training_args.temperature))
    logger.info("Dataset: %s", data_args.train_data)
    logger.info("Model: %s", model_args.model_name_or_path)
    logger.info("temperature: %s", training_args.temperature)
    
    trainer.train()
    
    logs = trainer.state.log_history
    with open(data_args.logging_pth + ".json", "w") as f:
        json.dump(logs, f, indent=2)

if __name__ == "__main__":
    main()
