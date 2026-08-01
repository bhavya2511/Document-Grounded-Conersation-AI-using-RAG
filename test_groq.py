import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # loads .env file

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

res = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=20
)

print(res.choices[0].message.content)