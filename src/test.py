from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI()

resp = client.responses.create(
    model="gpt-5-nano",
    input="Say hello in one word."
)

print(resp.output_text)