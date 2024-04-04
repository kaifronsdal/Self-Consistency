from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)
model_name = "deepseek-ai/deepseek-math-7b-instruct"
dtype = "float16"
llm = LLM(model=model_name, trust_remote_code=True, dtype=dtype, tensor_parallel_size=2)

while True:
    result = llm.generate(["What is 2+2?" * 100] * 100, sampling_params)