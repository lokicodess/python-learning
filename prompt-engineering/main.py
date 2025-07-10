from openai import OpenAI
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())


def get_completion():
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        temperature=0.3,  # more accuracy
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that speaks like Shakespeare.",
            },
            {"role": "user", "content": "tell me a joke"},
            {"role": "user", "content": "I don't know"},
        ],
    )
    return completion.choices[0].message


res = get_completion()
print(res)
