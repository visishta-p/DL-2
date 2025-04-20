import torch

def generate_lyrics(prompt, model_path="./output", max_length=100):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    sample_output = model.generate(
        input_ids,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        max_length=max_length,
        num_return_sequences=1
    )

    generated = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    return generated

# Example usage
if __name__ == "__main__":
    prompt = "I walk this lonely road"
    lyrics = generate_lyrics(prompt)
    print("🎶", lyrics)