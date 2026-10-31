# B2B Lead Enricher

A command-line tool that turns a company's website domain into sales-ready research. Give it a domain and it will:

1. **Validate and normalize** the input (`https://www.Acme.com/about` → `acme.com`)
2. **Fetch the homepage** and extract the readable text
3. **Infer 3–5 business pain points** with an OpenAI model
4. **Write 3 cold email drafts**, each from a different angle (problem-focused, social proof, question-led)

The model's response is checked against a strict [Pydantic](https://docs.pydantic.dev/) schema, so you always get a company summary, at least one pain point, and exactly three emails. If any step fails, you get a clear error message instead.

## Requirements

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Setup

```bash
git clone git@github.com:dlcellucci/b2b-lead-enricher-ai.git
cd b2b-lead-enricher-ai

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."
```

## Usage

```bash
python main.py <domain> [--sender "what you sell"] [--model MODEL] [--json]
```

| Option | Description |
|---|---|
| `domain` | Company domain or URL, e.g. `acme.com` or `https://www.acme.com/about` |
| `--sender` | One-line description of your offering, used to make the emails more relevant |
| `--model` | OpenAI model to use (default: `gpt-4o-mini`, or the `OPENAI_MODEL` environment variable) |
| `--json` | Print the result as JSON instead of readable text |

### Examples

Basic run:

```bash
python main.py stripe.com
```

Tailor the emails to your product:

```bash
python main.py https://www.acme.com --sender "We help manufacturers automate quoting with AI"
```

JSON output for piping into a CRM or other scripts:

```bash
python main.py acme.com --json > acme.json
```

Use a different model:

```bash
OPENAI_MODEL=gpt-4o python main.py acme.com
```

### Sample output

```text
=== Lead: acme.com ===

Acme manufactures industrial widgets for mid-size OEMs.

Pain points:
  1. Manual quoting — Custom orders require engineers to hand-build quotes, slowing sales cycles.
  2. Supply chain visibility — ...

--- Email draft 1 ---
Subject: Cutting quote turnaround at Acme

Hi there, ...

--- Email draft 2 ---
...
```

With `--json`:

```json
{
  "domain": "acme.com",
  "company_summary": "...",
  "pain_points": [{ "title": "...", "description": "..." }],
  "emails": [{ "subject": "...", "body": "..." }]
}
```

### Errors and exit codes

The tool exits with `0` on success and `1` on any expected failure. It prints an `Error: ...` message to stderr for:

| Error | Cause |
|---|---|
| `InvalidDomainError` | Empty, malformed, or non-string domain (e.g. `localhost`, `user@acme.com`, IP addresses) |
| `WebsiteFetchError` | Timeout, DNS/connection failure, HTTP 4xx/5xx, or a page with no readable text |
| `ConfigurationError` | `OPENAI_API_KEY` is not set |
| `LLMResponseError` | OpenAI API failure, empty or non-JSON reply, or a reply that fails schema validation (e.g. not exactly 3 emails) |

## Using it as a library

```python
from main import enrich_lead

result = enrich_lead("acme.com", sender_context="We sell CPQ software")
print(result.company_summary)
for email in result.emails:
    print(email.subject)
```

`enrich_lead` also accepts `client=` (an OpenAI client) and `session=` (a `requests.Session`) so you can inject your own or mock them in tests.

## Running tests

The test suite uses mocks for the OpenAI client and HTTP requests, so it needs **no API key and no network access**.

```bash
source venv/bin/activate
pytest            # quick summary
pytest -v         # every test listed by name
pytest -k domain  # only tests whose names match "domain"
```

The 54 tests cover:

- **Input validation**: normalization of URLs, ports, `www.`, and trailing dots; rejecting malformed domains and non-string input
- **Web fetching**: success, truncation of long pages, timeouts, connection errors, HTTP errors, empty pages
- **LLM handling**: API errors, empty or malformed replies, schema violations (wrong email count, missing fields, empty values)
- **CLI**: exit codes, error messages, and JSON or text output

## Project structure

```text
.
├── main.py            # Validation, scraping, LLM analysis, CLI
├── test_main.py       # pytest unit tests
├── requirements.txt   # openai, pydantic, requests, pytest
└── README.md
```

## Limitations

- Only the homepage is fetched, and pages that render their content with JavaScript may return little text.
- Pain points are inferred by an LLM from public website copy. Treat them as hypotheses and review every draft before sending.
