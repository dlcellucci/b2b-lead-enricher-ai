"""B2B Lead Enricher.

Takes a company website domain, scrapes the homepage, uses an LLM to infer the
company's likely business pain points, and generates 3 cold email drafts.

Usage:
    export OPENAI_API_KEY=sk-...
    python main.py example.com [--json] [--model MODEL] [--sender "We help X do Y"]
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from typing import Any

import requests
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
REQUEST_TIMEOUT = 10
MAX_PAGE_CHARS = 12_000
USER_AGENT = "Mozilla/5.0 (compatible; B2BLeadEnricher/1.0)"

DOMAIN_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$"
)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class LeadEnricherError(Exception):
    """Base class for all expected, user-facing errors."""


class InvalidDomainError(LeadEnricherError):
    pass


class WebsiteFetchError(LeadEnricherError):
    pass


class ConfigurationError(LeadEnricherError):
    pass


class LLMResponseError(LeadEnricherError):
    pass


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class PainPoint(BaseModel):
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)


class EmailDraft(BaseModel):
    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)


class EnrichmentResult(BaseModel):
    domain: str
    company_summary: str
    pain_points: list[PainPoint] = Field(min_length=1)
    emails: list[EmailDraft] = Field(min_length=3, max_length=3)


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def validate_domain(raw: Any) -> str:
    """Normalize and validate a domain. Accepts bare domains or full URLs.

    "https://www.Example.com/about?x=1" -> "example.com"
    """
    if not isinstance(raw, str):
        raise InvalidDomainError("Domain must be a string.")

    domain = raw.strip().lower()
    if not domain:
        raise InvalidDomainError("Domain must not be empty.")

    domain = re.sub(r"^[a-z][a-z0-9+.-]*://", "", domain)  # strip scheme
    domain = re.split(r"[/?#]", domain, maxsplit=1)[0]  # strip path/query
    if "@" in domain:
        raise InvalidDomainError(f"Invalid domain: {raw!r}")
    domain = domain.split(":", 1)[0]  # strip port
    domain = domain.rstrip(".")
    if domain.startswith("www."):
        domain = domain[4:]

    if not DOMAIN_RE.match(domain):
        raise InvalidDomainError(f"Invalid domain: {raw!r}")
    return domain


# --------------------------------------------------------------------------- #
# Website scraping
# --------------------------------------------------------------------------- #
def html_to_text(raw_html: str) -> str:
    """Very small HTML -> text conversion (no external parser needed)."""
    text = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", raw_html)
    text = re.sub(r"(?s)<!--.*?-->", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_website_text(domain: str, session: requests.Session | None = None) -> str:
    """Fetch the homepage of ``domain`` and return its visible text."""
    http = session or requests.Session()
    url = f"https://{domain}"
    try:
        resp = http.get(
            url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}
        )
        resp.raise_for_status()
    except requests.Timeout as exc:
        raise WebsiteFetchError(f"Timed out fetching {url}") from exc
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        raise WebsiteFetchError(f"{url} returned HTTP {status}") from exc
    except requests.RequestException as exc:
        raise WebsiteFetchError(f"Could not reach {url}: {exc}") from exc

    text = html_to_text(resp.text or "")
    if not text:
        raise WebsiteFetchError(f"No readable content found at {url}")
    return text[:MAX_PAGE_CHARS]


# --------------------------------------------------------------------------- #
# LLM analysis
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are a senior B2B sales researcher and copywriter.
Given the text of a company's website, you:
1. Summarize what the company does in 1-2 sentences.
2. Infer 3-5 concrete business pain points they likely face (operational,
   growth, technical, or market). Be specific to this company, not generic.
3. Write exactly 3 distinct cold email drafts addressed to a decision maker at
   that company. Each email: short (under 120 words), personalized, references
   one or more pain points, and ends with a low-friction call to action.
   Use different angles for each (e.g. problem-focused, social proof, question-led).

Respond ONLY with JSON in this exact shape:
{
  "company_summary": "string",
  "pain_points": [{"title": "string", "description": "string"}],
  "emails": [{"subject": "string", "body": "string"}]
}"""


def get_client() -> OpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        raise ConfigurationError("OPENAI_API_KEY environment variable is not set.")
    return OpenAI()


def analyze_company(
    domain: str,
    website_text: str,
    client: Any,
    model: str = DEFAULT_MODEL,
    sender_context: str | None = None,
) -> EnrichmentResult:
    """Ask the LLM for pain points + 3 email drafts and validate the response."""
    if not website_text or not website_text.strip():
        raise ValueError("website_text must not be empty.")

    user_msg = f"Company domain: {domain}\n\nWebsite text:\n{website_text}"
    if sender_context:
        user_msg += f"\n\nAbout the sender (what we offer): {sender_context}"

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
    except OpenAIError as exc:
        raise LLMResponseError(f"OpenAI API request failed: {exc}") from exc

    try:
        content = completion.choices[0].message.content
    except (AttributeError, IndexError) as exc:
        raise LLMResponseError("OpenAI returned an unexpected response shape.") from exc
    if not content:
        raise LLMResponseError("OpenAI returned an empty response.")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMResponseError("OpenAI response was not valid JSON.") from exc
    if not isinstance(data, dict):
        raise LLMResponseError("OpenAI response JSON was not an object.")

    try:
        return EnrichmentResult(domain=domain, **data)
    except (ValidationError, TypeError) as exc:
        raise LLMResponseError(f"OpenAI response failed validation: {exc}") from exc


# --------------------------------------------------------------------------- #
# Orchestration / CLI
# --------------------------------------------------------------------------- #
def enrich_lead(
    raw_domain: str,
    client: Any | None = None,
    session: requests.Session | None = None,
    model: str = DEFAULT_MODEL,
    sender_context: str | None = None,
) -> EnrichmentResult:
    domain = validate_domain(raw_domain)
    text = fetch_website_text(domain, session=session)
    return analyze_company(
        domain,
        text,
        client=client or get_client(),
        model=model,
        sender_context=sender_context,
    )


def format_result(result: EnrichmentResult) -> str:
    lines = [f"=== Lead: {result.domain} ===", "", result.company_summary, ""]
    lines.append("Pain points:")
    for i, p in enumerate(result.pain_points, 1):
        lines.append(f"  {i}. {p.title} — {p.description}")
    for i, e in enumerate(result.emails, 1):
        lines += ["", f"--- Email draft {i} ---", f"Subject: {e.subject}", "", e.body]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B2B Lead Enricher")
    parser.add_argument("domain", help="Company website domain, e.g. acme.com")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--sender", help="Short description of what you sell")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args(argv)

    try:
        result = enrich_lead(args.domain, model=args.model, sender_context=args.sender)
    except LeadEnricherError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(result.model_dump_json(indent=2) if args.json else format_result(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
