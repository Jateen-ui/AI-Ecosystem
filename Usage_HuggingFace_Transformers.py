# Excellent Code Works very well, shows how to use different LLM Models

from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv
from huggingface_hub import login
import os
import torch

# 1. Load HuggingFace token
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
login(token=hf_token)

def load_chatbot(model_name, use_4bit=True):
    """
    Universal loader for any chat LLM. Works for:
    Mistral, Llama-3.1, Phi-3, Gemma-2, Qwen2, DeepSeek

    Args:
        model_name: HF repo like "mistralai/Mistral-7B-Instruct-v0.3"
        use_4bit: Set False if you have 24GB+ VRAM
    """
    # 1. 4-bit config = no more OOM errors
    if use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model_kwargs = {"quantization_config": bnb_config}
    else:
        model_kwargs = {}

    # 2. Tokenizer handles chat template automatically
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 3. Pipeline handles everything else
    chatbot = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        model_kwargs=model_kwargs,
        device_map="auto", # CPU if no GPU
        torch_dtype="auto", # FP16/BF16 auto
        trust_remote_code=True # Needed for Phi, Qwen, DeepSeek
    )
    return chatbot

def ask(chatbot, question, system_prompt="You are a helpful assistant."):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    output = chatbot(
        messages,
        max_new_tokens=200,
        do_sample=False, # Fast + deterministic
        return_full_text=False
    )
    return output[0]['generated_text']

# ------------------- USAGE -------------------
# Just change the model name. Everything else stays same.

# Example 1: Mistral - Apache 2.0, 32K context
bot = load_chatbot("mistralai/Mistral-7B-Instruct-v0.3")
print(ask(bot, "What is Problem Management Process"))

# Example 2: Phi-3-mini - MIT, super fast, 2GB VRAM
# bot = load_chatbot("microsoft/Phi-3-mini-4k-instruct")
# print(ask(bot, "Who are you?", system_prompt="You are a pirate"))

# Example 3: Gemma-2-9B - Apache 2.0, strong quality
# bot = load_chatbot("google/gemma-2-9b-it")

# Example 4: Llama-3.1-8B - 128K context
# bot = load_chatbot("meta-llama/Llama-3.1-8B-Instruct")

# Example 5: Qwen2-7B - Best multi-lingual
# bot = load_chatbot("Qwen/Qwen2-7B-Instruct")