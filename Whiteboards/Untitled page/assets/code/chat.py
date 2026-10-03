"""
Groq CLI Chatbot — stateless (no memory) + streaming.
Each question is sent on its own; the model does not remember earlier turns.

Run:  python chat.py
Commands:  /exit
"""

from groq import Groq

import config

client = Groq(api_key=config.GROQ_API_KEY)


def ask(user_input):
    """Send a single question to Groq and stream the reply."""
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    stream = client.chat.completions.create(
        model=config.MODEL,
        messages=messages,
        max_tokens=config.MAX_TOKENS,
        temperature=config.TEMPERATURE,
        stream=True,
    )
    print("AI: ", end="", flush=True)
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
    print("\n")


def main():
    print(f"🤖 Groq Chat (no memory) | Model: {config.MODEL}")
    print("Type /exit to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            print("👋 Bye!")
            break

        try:
            ask(user_input)
        except Exception as e:
            print(f"\n⚠️ Error: {e}\n")


if __name__ == "__main__":
    main()