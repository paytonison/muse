#!/usr/bin/env python3
"""
michaelangelo_chat.py – talk to your Michaelangelo fine-tune from the CLI
"""

import os, sys, json
from openai import OpenAI

# -------------------------------------------------------------------
# *** FILL THESE IN **************************************************
MODEL_ID       = "ft:gpt-4.1-mini-2025-04-14:the-singularity:what-the-world-has-come-to:BjFMgKBn"   # ← your fine-tuned model
PROMPT_ID      = "pmpt_6850db82c14881958b901520c9743c6b0e0f7094ce67868c"
PROMPT_VERSION = "4"
# -------------------------------------------------------------------

client = OpenAI()   # assumes OPENAI_API_KEY is set in the environment

conversation = [
    {
        "role": "user",
        "content": [{"type": "input_text", "text": "Hello. What is your name?"}]
    }
]

def call_michaelangelo(conv):
    """Send conversation, return assistant message dict."""
    resp = client.responses.create(
        model   = MODEL_ID,
        prompt  = {"id": PROMPT_ID, "version": PROMPT_VERSION},
        input   = conv,
        reasoning = {},
        temperature = 0.85,
        top_p = 0.9,
        max_output_tokens = 2048,
        store   = True
    )
    # For the responses endpoint the assistant message is always the *last*
    # element of the returned `output` list.
    return resp.output[-1]

def main():
    print("Michaelangelo is ready – type messages, 'exit' to quit\n")
    try:
        while True:
            user = input("You  > ").strip()
            if user.lower() in {"exit", "quit", "q"}:
                break

            conversation.append({
                "role": "user",
                "content": [{"type": "input_text", "text": user}]
            })

            assistant_msg = call_michaelangelo(conversation)
            conversation.append(assistant_msg.model_dump(exclude_none=True))

            print("Michaelangelo >", assistant_msg.content[0].text, "\n")

    except (KeyboardInterrupt, EOFError):
        print("\n[Session ended]")

    # Optional local log
    # with open("muse_log.json", "w") as fp:
    #     json.dump(conversation, fp, indent=2)

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first.")
    main()
