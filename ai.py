
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("AI_MODEL", "openai/gpt-oss-120b")
MAX_HISTORY = 12

# Temporary memory, stored in RAM.
conversations = {}

PERSONALITIES = {
    "companion": (
        "You are a friendly AI companion. "
        "Talk naturally, be supportive, and be honest "
        "that you are an AI."
    ),
    "tutor": (
        "You are an adaptive learning tutor. "
        "Explain concepts step by step, give examples, "
        "and check the learner's understanding."
    ),
    "both": (
        "You are a friendly AI companion and adaptive tutor. "
        "Talk naturally, and teach clearly when asked. "
        "Adjust explanations to the learner's level."
    ),
}
#Fungsi Response
def generate_response(
    message: str,
    user_id: str,
    mode: str,
    learning_level: str,
) -> str:

    api_key = os.getenv("AI_API_KEY")

    if not api_key:
        raise RuntimeError("AI_API_KEY is missing.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    instructions = (
        f"{PERSONALITIES[mode]}\n"
        f"Learner level: {learning_level}.\n"
        "Use the user's language when possible. "
        "Explain difficult concepts clearly. "
        "Do not invent memories or facts."
    )

    history = conversations.get(user_id, [])

    messages = [
        {"role": "system", "content": instructions},
        *history,
        {"role": "user", "content": message},
    ]

    # Groq-compatible Chat Completions API.
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    reply = response.choices[0].message.content

    if not reply or not reply.strip():
        raise RuntimeError("The AI returned an empty response.")

    # Save the conversation only after success.
    conversations[user_id] = (
        history
        + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ]
    )[-MAX_HISTORY:]

    return reply
