#!/usr/bin/env python3
"""
Vext-labs-7B-v1.1 — Inference Script
Run autonomous penetration testing analysis with a single command.

Usage:
    python run.py --prompt "Analyze this nmap scan: ..."
    python run.py --prompt-file scan_output.txt
    python run.py --interactive
"""

import argparse
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "Vext-Labs-Inc/Vext-labs-7B-v1.1-"


def load_model(device_map="auto", dtype=torch.bfloat16):
    """Load the model and tokenizer."""
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=device_map,
    )
    print("Model loaded successfully.")
    return tokenizer, model


def generate(tokenizer, model, prompt, max_new_tokens=512, temperature=0.7):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    # Decode only the new tokens (skip the prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def interactive_mode(tokenizer, model, args):
    """Run an interactive session."""
    print("\n" + "=" * 60)
    print("  VEXT-labs-7B-v1.1 — Interactive Mode")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue

        response = generate(
            tokenizer, model, prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"\n{response}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with Vext-labs-7B-v1.1"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Text prompt to send to the model"
    )
    parser.add_argument(
        "--prompt-file", type=str, default=None,
        help="Path to a file containing the prompt (e.g., scan output)"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Launch interactive chat mode"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Maximum new tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7, use 0 for greedy)"
    )
    parser.add_argument(
        "--device-map", type=str, default="auto",
        help="Device map for model loading (default: auto)"
    )
    args = parser.parse_args()

    if not args.prompt and not args.prompt_file and not args.interactive:
        parser.print_help()
        print("\nError: provide --prompt, --prompt-file, or --interactive")
        sys.exit(1)

    tokenizer, model = load_model(device_map=args.device_map)

    if args.interactive:
        interactive_mode(tokenizer, model, args)
        return

    # Get prompt from argument or file
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompt = f.read().strip()
    else:
        prompt = args.prompt

    response = generate(
        tokenizer, model, prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(response)


if __name__ == "__main__":
    main()
