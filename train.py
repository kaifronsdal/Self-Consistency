import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7,8,9"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "128"

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


from pathlib import Path

from torch.utils.data import Dataset, DataLoader
import torch
import transformers
import json
import logging
import copy
from dataclasses import dataclass
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import random_split
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import set_seed
from torch.optim import AdamW

from peft import PromptTuningConfig, PromptTuningInit, get_peft_model

from prompt import make_full_source, make_full_target, make_answer_only_source, make_answer_only_target

from util import save_output


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
                source = make_full_source(example)
                target = make_full_target(example)
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

def training_loop(root_dir, data_args, lr, num_epochs):
    set_seed(0)
    accelerator = Accelerator(mixed_precision="bf16")

    model_name = "deepseek-ai/deepseek-math-7b-instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    # get model max input size
    model_max_length = model.config.max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=model_max_length,
                                      padding_side="right")

    tokenizer.add_special_tokens(
        {
            "bos_token": DEFAULT_BOS_TOKEN,
            "eos_token": DEFAULT_EOS_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )

    prompt_tuning_init_text = "Classify if the tweet is a complaint or no complaint.\n"
    peft_config = PromptTuningConfig(
        task_type="CAUSAL_LM",
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=len(tokenizer(prompt_tuning_init_text)["input_ids"]),
        prompt_tuning_init_text=prompt_tuning_init_text,
        tokenizer_name_or_path=model_name,
    )
    
    model = get_peft_model(model, peft_config)


    data_module = make_supervised_data_module(tokenizer, data_args, load=True, override=False,
                                      output_path=root_dir / "data/selfee-train_preprocessed.pkl")

    train_dataloader = DataLoader(data_module["train_dataset"], shuffle=True, collate_fn=_collate_fn,
                          batch_size=1, pin_memory=True)
    eval_dataloader = DataLoader(data_module["eval_dataset"], collate_fn=_collate_fn, batch_size=1,
                         pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
        model.eval()
        eval_loss = 0
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.item()
    
        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

if __name__ == "__main__":
    root_dir = Path("~").expanduser()
    
    data_args = {
        "data_path": root_dir / "data/selfee-train.json",
        "training_objective": "full",
    }

    lr = 3e-2
    num_epochs = 50
    
    training_loop(root_dir, data_args, lr, num_epochs)
