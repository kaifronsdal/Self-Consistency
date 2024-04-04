import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4,6,7,8,9"
# CUDA_VISIBLE_DEVICES="1,3,5" accelerate launch --config_file accelerate_config3.yaml finetune.py

IGNORE_INDEX = -100
# DEFAULT_PAD_TOKEN = "[PAD]"
# DEFAULT_EOS_TOKEN = "</s>"
# DEFAULT_BOS_TOKEN = "</s>"
# DEFAULT_UNK_TOKEN = "</s>"

from torch.utils.data import Dataset, DataLoader
import torch
import transformers
import json
import logging
import copy

from torch.utils.data import random_split
from transformers import TrainingArguments, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from peft import get_peft_model, LoraConfig

from tqdm import tqdm

from prompt import make_full_source, make_full_target, make_answer_only_source, make_answer_only_target, \
    make_full_source_from_template, make_full_target_from_template

from util import save_output

from pathlib import Path


def _tokenize_fn(strings: list[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in tqdm(strings)
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: list[str],
        targets: list[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    if isinstance(sources, list):
        examples = [tokenizer.apply_chat_template(s, tokenize=False, add_generation_prompt=True) for s in examples]
        sources = [tokenizer.apply_chat_template(s, tokenize=False, add_generation_prompt=True) for s in sources]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in tqdm(zip(labels, sources_tokenized["input_ids_lens"])):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, training_objective: str):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        # list_data_dict = json.load(open(data_path, "r"))
        list_data_dict = []
        with open(data_path, "r") as f:
            for line in f:
                list_data_dict.append(json.loads(line))

        sources = []
        targets = []
        logging.warning("Formatting inputs...")
        if training_objective == "full":
            for example in list_data_dict:
                source = make_full_source_from_template(example)
                target = make_full_target_from_template(example)
                target += f"{tokenizer.eos_token}"
                sources.append(source)
                targets.append(target)
        elif training_objective == "answer-only":
            for example in list_data_dict:
                source = make_answer_only_source(example)
                target = make_answer_only_target(example, eos_token=tokenizer.eos_token)
                sources.append(source)
                targets.append(target)
        else:
            raise NotImplementedError

        logging.warning(f"Number of source examples: {len(sources)}")
        logging.warning(f"Number of target examples: {len(targets)}")

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@save_output
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer,
                                      data_path=data_args['data_path'],
                                      training_objective=data_args['training_objective'])
    train_dataset, eval_dataset = random_split(train_dataset, [int(len(train_dataset) * 0.9),
                                                               len(train_dataset) - int(len(train_dataset) * 0.9)])

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


def train(model_name, lr, num_epochs, data_args=None, root_dir=None):
    if data_args is None:
        data_args = {
            "data_path": "data/selfee-train.json",
            "training_objective": "full",
        }
    if root_dir is None:
        root_dir = Path(".")

    # check to see if we should load from a checkpoint
    checkpoint_dir = root_dir / "checkpoint_lora"

    last_checkpoint = None
    if checkpoint_dir.exists():
        last_checkpoint = get_last_checkpoint(checkpoint_dir)

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        eval_steps=1000,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=4,
        warmup_steps=0.03,
        learning_rate=lr,
        fp16=True,
        report_to="none",
        seed=42,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto')
    # get model max input size
    model_max_length = model.config.max_position_embeddings
    model_max_length = 2048
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=model_max_length,
                                              padding_side="right")
    # tokenizer.add_special_tokens(
    #     # {
    #     #     "bos_token": DEFAULT_BOS_TOKEN,
    #     #     "eos_token": DEFAULT_EOS_TOKEN,
    #     #     "pad_token": DEFAULT_PAD_TOKEN,
    #     #     "unk_token": DEFAULT_UNK_TOKEN,
    #     # }
    # )
    tokenizer.pad_token = tokenizer.eos_token

    data_module = make_supervised_data_module(tokenizer, data_args, load=True, override=False,
                                              output_path=root_dir / "data/selfee-train_preprocessed.pkl")

    def _collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        input_ids, labels = tuple([example[key] for example in batch]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )


    # train_dataloader = DataLoader(data_module["train_dataset"], shuffle=True, collate_fn=_collate_fn,
    #                               batch_size=1, pin_memory=True)
    # eval_dataloader = DataLoader(data_module["eval_dataset"], collate_fn=_collate_fn, batch_size=1,
    #                              pin_memory=True)

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    max_seq_length = 2048
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        # max_seq_length=max_seq_length,
        data_collator=_collate_fn,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
    )

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()
    trainer.save_state()

    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # lr_scheduler = get_linear_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=(len(train_dataloader) * num_epochs),
    # )

    # args = TrainingArguments(
    #     output_dir="mistral_instruct_generation",
    #     # num_train_epochs=5,
    #     max_steps=100,
    #     per_device_train_batch_size=4,
    #     warmup_steps=0.03,
    #     logging_steps=10,
    #     save_strategy="epoch",
    #     # evaluation_strategy="epoch",
    #     evaluation_strategy="steps",
    #     eval_steps=20,
    #     learning_rate=2e-4,
    #     bf16=True,
    # )


def main():
    # model_name = "deepseek-ai/deepseek-math-7b-instruct"
    # model_name = "google/gemma-2b-it"
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    root_dir = Path("~").expanduser()

    data_args = {
        "data_path": root_dir / "data/selfee-train.json",
        "training_objective": "full",
    }

    lr = 1e-5
    num_epochs = 4

    train(model_name, lr, num_epochs, data_args=data_args, root_dir=root_dir)


if __name__ == "__main__":
    main()
