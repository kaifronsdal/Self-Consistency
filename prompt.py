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
        [f'{i + 1}. {ex["question"]}\n\n{ex["answer"]}\n\n' for i, ex in enumerate(prompt_examples)])
    # prompt_header += '\n'.join([f'{ex["question"]}\n\n{ex["answer"]}\n\n' for ex in prompt_examples])

    return [{
        'question': prompt_header + f'\n{few_shot_count + 1}. {ex["question"]}',
        'answer': ex["answer"],
        'boxed': ex["boxed"]
    } for ex in remaining_examples]
