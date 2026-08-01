import os
import time
import logging
from typing import List
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()
logger = logging.getLogger(__name__)

# Initialize client (NEW SDK STYLE)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Models
LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")


# =========================
# LLM RESPONSE
# =========================
def get_llm_response(prompt: str, max_tokens: int = 2048) -> str:
    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=max_tokens,
            )
        )
        return response.text.strip()

    except Exception as e:
        logger.warning(f"Gemini failed: {e}")
        return _groq_fallback(prompt, max_tokens)


# =========================
# EMBEDDINGS (DOCUMENT)
# =========================
def get_embedding(text: str, max_retries: int = 3) -> List[float]:
    text = text[:8000]

    # NEW FORMAT (IMPORTANT)
    formatted_text = f"title: none | text: {text}"

    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=formatted_text,
                config=types.EmbedContentConfig(
                    output_dimensionality=768
                )
            )

            return result.embeddings[0].values

        except Exception as e:
            logger.warning(f"Embedding attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


# =========================
# EMBEDDINGS (QUERY)
# =========================
def get_query_embedding(text: str, max_retries: int = 3) -> List[float]:
    text = text[:2000]

    # NEW QUERY FORMAT
    formatted_text = f"task: search result | query: {text}"

    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=formatted_text,
                config=types.EmbedContentConfig(
                    output_dimensionality=768
                )
            )

            return result.embeddings[0].values

        except Exception as e:
            logger.warning(f"Query embedding attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise

# =========================
# GROQ FALLBACK
# =========================
def _groq_fallback(prompt: str, max_tokens: int = 2048) -> str:
    groq_key = os.getenv("GROQ_API_KEY")

    if not groq_key:
        logger.error("No Groq API key")
        return ""

    try:
        from groq import Groq
        import httpx

        logger.info("Using Groq fallback...")

        client_groq = Groq(
            api_key=groq_key,
            http_client=httpx.Client()  # ← FORCE CLEAN CLIENT
        )

        response = client_groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens
        )

        text = response.choices[0].message.content.strip()

        logger.info(f"Groq success, length={len(text)}")

        return text

    except Exception as e:
        logger.error(f"Groq fallback failed: {e}")
        return ""