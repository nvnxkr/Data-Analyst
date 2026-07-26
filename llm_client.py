"""
llm_client.py
Thin wrapper around the Mistral chat completions API.
Handles: NL question -> structured JSON intent (pandas code + chart spec),
and automatic insight generation. Fails gracefully on timeout/error.
"""

import os
import json
import re
import requests

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-large-latest")
REQUEST_TIMEOUT = 30


class LLMError(Exception):
    pass


def _get_api_key() -> str:
    key = os.environ.get("MISTRAL_API_KEY", "")
    if not key:
        raise LLMError(
            "MISTRAL_API_KEY is not set. Add it to your .env file (see .env.example)."
        )
    return key


def _call_mistral(messages, temperature=0.2, response_format_json=False):
    api_key = _get_api_key()
    payload = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}

    try:
        resp = requests.post(
            MISTRAL_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.Timeout:
        raise LLMError("The AI request timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        raise LLMError(f"Could not reach the Mistral API: {e}")

    if resp.status_code != 200:
        raise LLMError(f"Mistral API error ({resp.status_code}): {resp.text[:300]}")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise LLMError("Unexpected response shape from Mistral API.")


def _extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from the model's reply."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise LLMError("Could not parse a JSON response from the AI.")


QA_SYSTEM_PROMPT = """You are a data analysis assistant. You are given a pandas
DataFrame's schema and a sample of its rows, plus a user's natural-language
question. Respond with ONLY a JSON object (no markdown, no commentary) with
these keys:

{
  "explanation": "one or two sentence plain-English answer/description",
  "pandas_code": "python code using only the variable `df` (already loaded)
                   and `pd`/`np`; must assign the final answer to a variable
                   named `result`. Do not import anything. Do not define
                   functions. Keep it to simple, safe pandas operations
                   (filtering, groupby, sort_values, head, aggregation, etc.)",
  "chart_type": "one of: bar, line, pie, scatter, none",
  "chart_x": "column name to use for x-axis / categories, or null",
  "chart_y": "column name to use for y-axis / values, or null"
}

Rules:
- `result` should be a DataFrame or Series suitable for display and charting.
- Never use eval, exec, open, import, or file I/O.
- If the question cannot be answered from the given schema, set pandas_code
  to `result = df.head(10)` and explain why in `explanation`.
"""


def get_query_intent(question: str, schema_text: str) -> dict:
    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Schema and sample data:\n{schema_text}\n\nQuestion: {question}",
        },
    ]
    content = _call_mistral(messages, temperature=0.1, response_format_json=True)
    intent = _extract_json(content)
    for key in ("pandas_code", "chart_type"):
        if key not in intent:
            raise LLMError(f"AI response missing required field: {key}")
    return intent


INSIGHTS_SYSTEM_PROMPT = """You are a data analyst. Given a dataset's schema,
summary statistics, and a sample of rows, produce 3-5 short, concrete bullet
point insights (trends, outliers, top/bottom performers, notable
correlations). Respond with ONLY a JSON object:

{ "insights": ["insight 1", "insight 2", ...] }

Keep each insight to one sentence. Be specific and reference actual column
names/values where possible. Do not invent data that isn't implied by the
summary provided.
"""


def get_auto_insights(schema_text: str, summary: dict) -> list:
    messages = [
        {"role": "system", "content": INSIGHTS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Schema and sample:\n{schema_text}\n\n"
                f"Summary stats (JSON):\n{json.dumps(summary, default=str)}"
            ),
        },
    ]
    content = _call_mistral(messages, temperature=0.4, response_format_json=True)
    parsed = _extract_json(content)
    insights = parsed.get("insights", [])
    if not insights:
        raise LLMError("AI did not return any insights.")
    return insights
