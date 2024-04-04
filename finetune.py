import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4,6,7,8,9"

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
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from accelerate import Accelerator
from accelerate.utils import set_seed

from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, LoraConfig

from tqdm import tqdm

from prompt import make_full_source, make_full_target, make_answer_only_source, make_answer_only_target

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


def train(model_name, lr, num_epochs, mixed_precision='fp16', data_args=None, root_dir=None):
    if data_args is None:
        data_args = {
            "data_path": "data/selfee-train.json",
            "training_objective": "full",
        }
    if root_dir is None:
        root_dir = Path(".")

    set_seed(42)
    accelerator = Accelerator(mixed_precision=mixed_precision)

    # check to see if we should load from a checkpoint
    checkpoint_dir = root_dir / "checkpoint"
    # if checkpoint_dir.exists() and any(checkpoint_dir.glob("checkpoint_*")):
    #     # load most recent checkpoint
    #     print("Loading most recent checkpoint.")
    #     checkpoint = sorted(checkpoint_dir.glob("checkpoint_*"))[-1]
    #     print(f"Loading checkpoint from {checkpoint}.")
    #     model = AutoModelForCausalLM.from_pretrained(checkpoint)
    # else:
    #     model = AutoModelForCausalLM.from_pretrained(model_name)

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

    train_dataloader = DataLoader(data_module["train_dataset"], shuffle=True, collate_fn=_collate_fn,
                                  batch_size=1, pin_memory=True)
    eval_dataloader = DataLoader(data_module["eval_dataset"], collate_fn=_collate_fn, batch_size=1,
                                 pin_memory=True)

    # prompt_tuning_init_text = "Is this response correct or not? Provide feedback explaining your reasoning first."
    # peft_config = PromptTuningConfig(
    #     task_type="CAUSAL_LM",
    #     prompt_tuning_init=PromptTuningInit.TEXT,
    #     num_virtual_tokens=len(tokenizer(prompt_tuning_init_text)["input_ids"]),
    #     prompt_tuning_init_text=prompt_tuning_init_text,
    #     tokenizer_name_or_path=model_name,
    # )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # check for checkpoint
    if checkpoint_dir.exists() and any(checkpoint_dir.glob("save_state_*")):
        # load most recent checkpoint
        print("Loading most recent checkpoint.")
        checkpoint = sorted(checkpoint_dir.glob("save_state_*"))[-1]
        print(f"Loading checkpoint from {checkpoint}.")
        accelerator.load_state(checkpoint)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            outputs = model(**batch)
            loss = outputs.loss
            # if loss becomes NaN
            if not torch.isfinite(loss).item():
                print(f"Loss is {loss}.")
                print("Skipping step.")
                continue

            train_loss += accelerator.gather(loss).detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % 1000 == 0:
                accelerator.print(f"{epoch=}: {step=}/{len(train_dataloader)}: {train_loss / (step + 1)} loss")

            if step % 2000 == 0:
                # save checkpoint
                accelerator.save_state(checkpoint_dir / f"save_state_{epoch}_{step}")
                # accelerator.free_memory()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    checkpoint_dir / f"checkpoint_{epoch}_{step}",
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                )

        model.eval()
        eval_loss = 0
        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += accelerator.gather(loss).detach().float()

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = train_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

        # save checkpoint
        # accelerator.save_state(checkpoint_dir / f"checkpoint_{epoch}")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            checkpoint_dir / f"checkpoint_{epoch}",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )

    # save model
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        checkpoint_dir / "final_model",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )


def main():
    # model_name = "deepseek-ai/deepseek-math-7b-instruct"
    model_name = "deepseek-ai/deepseek-math-7b-instruct"

    root_dir = Path("~").expanduser()

    data_args = {
        "data_path": root_dir / "data/selfee-train.json",
        "training_objective": "full",
    }

    lr = 2e-4
    num_epochs = 10

    train(model_name, lr, num_epochs, data_args=data_args, root_dir=root_dir)


if __name__ == "__main__":
    main()
