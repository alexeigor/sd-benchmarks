import os
os.environ['HF_HOME']='/workspace/.cache/huggingface'

import torch

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

def main():
    tokenizer = AutoTokenizer.from_pretrained("daryl149/llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained("daryl149/llama-2-7b-chat-hf", torch_dtype=torch.float16)

    print(model)

    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in Llama-2-13B: {num_parameters}")


if __name__ == "__main__":
    main()
