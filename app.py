import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline(
 "text-generation", model=model_id, max_length=200, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

messages = [
    {"role": "system", "content": "You are a helpful assistant!"},
    {"role": "user", "content": """Generate an approximately fifteen-word sentence 
                                   that describes all this data:
                                   Midsummer House eatType restaurant; 
                                   Midsummer House food Chinese; 
                                   Midsummer House priceRange moderate; 
                                   Midsummer House customer rating 3 out of 5; 
                                   Midsummer House near All Bar One"""},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

print(outputs[0]["generated_text"][len(prompt):])

