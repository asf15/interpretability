from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

prompt = "The most interesting thing about language models is"

inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
