def generate_nshot_prompts(dataset, n: int | list[int] = 5):
    """
    Generate n-shot prompts for a given dataset.

    :param dataset: A list of dictionaries, each containing a question and answer.
    :param n: The number of examples to include in the prompt. If an integer, the first n examples will be included.
              If a list, the examples at the given indices will be included.
    :return: A list of (prompt, answer) tuples.
    """
    n = list(range(n)) if isinstance(n, int) else n
    few_shot_count = len(n)
    prompt_examples = [dataset[i] for i in n]
    remaining_examples = [ex for i, ex in enumerate(dataset) if i not in n]

    prompt_header = f'Answer the following {few_shot_count + 1} questions:\n\n'
    prompt_header += '\n'.join(
        [f'{i + 1}. {ex["question"]}\n\nANSWER: {ex["answer"]}\n\n' for i, ex in enumerate(prompt_examples)])
    # prompt_header += '\n'.join([f'{ex["question"]}\n\n{ex["answer"]}\n\n' for ex in prompt_examples])

    return [{
        'question': prompt_header + f'\n{few_shot_count + 1}. {ex["question"]}\n\nANSWER: ',
        'answer': ex["answer"],
        'boxed': ex["boxed"]
    } for ex in remaining_examples]


def generate_nshot_prompts_from_template(dataset, n: int | list[int] = 5):
    """
    Generate n-shot prompts for a given dataset. Uses huggingface tokenizer to generate prompt template like
    messages = [
        {"role": "user", "content": "Who are you? Please, answer in pirate-speak."},
    ]
    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    :param dataset: A list of dictionaries, each containing a question and answer.
    :param n: The number of examples to include in the prompt. If an integer, the first n examples will be included.
              If a list, the examples at the given indices will be included.
    :return: A list of (prompt, answer) tuples.
    """
    n = list(range(n)) if isinstance(n, int) else n
    prompt_examples = [dataset[i] for i in n]
    remaining_examples = [ex for i, ex in enumerate(dataset) if i not in n]

    # question_modifier = "\n\nWrap your final answer with the `\\boxed{}` command."
    question_modifier = ""

    few_shot_messages = [
        {"role": role, "content": content}
        for ex in prompt_examples for role, content in
        zip(["user", "assistant", "user", "assistant"],
            [ex["question"] + question_modifier, ex["answer"], "FINAL ANSWER: ", ex["boxed"]])
    ]

    return [{
        'question':
            few_shot_messages + [{"role": "user", "content": ex["question"] + question_modifier}]
            + [{"role": "user", "content": "FINAL ANSWER: "}],
        'answer': ex["answer"],
        'boxed': ex["boxed"]
    } for ex in remaining_examples]


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Answer:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Answer:\n"
    ),
}
ANSWER_TEXT = "{output}"
REVISION_TEXT = "\n\n### Revision {number}:\n{revision}"


def make_full_source(example):
    """Make prompt for self-feedback."""

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    prompt = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
    return prompt


def make_full_target(example):
    """Make target for self-feedback."""
    prompt = ""
    outputs = example["outputs"]
    for i, output in enumerate(outputs):
        if i == 0:
            answer = output["output"]
            prompt += ANSWER_TEXT.format(output=answer)
        else:
            revision = output["output"]
            revision_number = i
            prompt += REVISION_TEXT.format(number=revision_number, revision=revision)
        if "feedback" in output:
            feedback = output["feedback"]
            feedback_number = i + 1
            prompt += "\n\n### Feedback {number}:\n{feedback}".format(number=feedback_number, feedback=feedback)

    return prompt


def make_answer_only_source(example):
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    prompt = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)

    return prompt


def make_answer_only_target(example, eos_token):
    answer = example["outputs"][0]["output"]
    prompt = "{output}".format(output=answer)

    return prompt + eos_token


if __name__ == "__main__":
    example_len2_model1 = {
        "instruction": "Write a response that appropriately completes the request.",
        "input": "This is the input.",
        "outputs": [
            {
                "output": "This is the first model output.",
                "feedback": "This is the chatGPT feedback to the first model output.",
            },
            {
                "output": "This is the second chatGPT output.",
                "feedback": "This is the chatGPT feedback to the second chatGPT output.",
            },
        ],
        "model-output-length": 1,
    }

    assert make_answer_only_source(example_len2_model1) == (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nWrite a response that appropriately completes the request.\n\n"
        "### Input:\nThis is the input.\n\n"
        "### Answer:\n"
    )
    assert make_answer_only_target(example_len2_model1, eos_token="</s>") == (
        "This is the first model output.</s>"
    )
