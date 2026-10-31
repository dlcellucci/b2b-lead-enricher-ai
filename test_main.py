import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests
from openai import OpenAIError

import main
from main import (
    ConfigurationError,
    EnrichmentResult,
    InvalidDomainError,
    LLMResponseError,
    WebsiteFetchError,
    analyze_company,
    enrich_lead,
    fetch_website_text,
    html_to_text,
    validate_domain,
)

VALID_PAYLOAD = {
    "company_summary": "Acme sells widgets to manufacturers.",
    "pain_points": [
        {"title": "Supply chain delays", "description": "Lead times are long."},
        {"title": "Manual quoting", "description": "Quotes take days."},
    ],
    "emails": [
        {"subject": "Faster quotes?", "body": "Hi there, ..."},
        {"subject": "How Beta cut lead times", "body": "Hi there, ..."},
        {"subject": "Quick question", "body": "Hi there, ..."},
    ],
}


def make_client(content=None, exc=None):
    """Fake OpenAI client whose chat.completions.create returns `content`."""
    client = MagicMock()
    if exc is not None:
        client.chat.completions.create.side_effect = exc
    else:
        msg = SimpleNamespace(content=content)
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=msg)]
        )
    return client


def make_session(text="<html><body><h1>Acme</h1><p>Widgets</p></body></html>",
                 status=200, exc=None):
    session = MagicMock()
    if exc is not None:
        session.get.side_effect = exc
        return session
    resp = MagicMock()
    resp.text = text
    resp.status_code = status
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    session.get.return_value = resp
    return session


# --------------------------------------------------------------------------- #
# validate_domain
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("example.com", "example.com"),
        ("  Example.COM  ", "example.com"),
        ("www.example.com", "example.com"),
        ("https://www.example.com/about?x=1#top", "example.com"),
        ("http://sub.example.co.uk:8080/", "sub.example.co.uk"),
        ("example.com.", "example.com"),
        ("my-company.io", "my-company.io"),
    ],
)
def test_validate_domain_accepts_and_normalizes(raw, expected):
    assert validate_domain(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "localhost",
        "not a domain",
        "-bad.com",
        "bad-.com",
        "exa_mple.com",
        "example",
        "example.c",
        "user@example.com",
        "http://",
        "a" * 64 + ".com",
        "192.168.1.1",
    ],
)
def test_validate_domain_rejects_invalid(raw):
    with pytest.raises(InvalidDomainError):
        validate_domain(raw)


@pytest.mark.parametrize("raw", [None, 123, ["example.com"], b"example.com"])
def test_validate_domain_rejects_non_strings(raw):
    with pytest.raises(InvalidDomainError, match="string"):
        validate_domain(raw)


# --------------------------------------------------------------------------- #
# html_to_text / fetch_website_text
# --------------------------------------------------------------------------- #
def test_html_to_text_strips_scripts_styles_and_tags():
    raw = (
        "<html><head><style>.x{}</style><script>alert(1)</script></head>"
        "<body><!-- hidden --><p>Hello &amp; welcome</p>  <div>World</div></body></html>"
    )
    assert html_to_text(raw) == "Hello & welcome World"


def test_fetch_website_text_success():
    session = make_session()
    assert fetch_website_text("acme.com", session=session) == "Acme Widgets"
    assert session.get.call_args.args[0] == "https://acme.com"


def test_fetch_website_text_truncates_long_pages():
    session = make_session(text="<p>" + "a" * (main.MAX_PAGE_CHARS + 500) + "</p>")
    assert len(fetch_website_text("acme.com", session=session)) == main.MAX_PAGE_CHARS


def test_fetch_website_text_timeout():
    session = make_session(exc=requests.Timeout())
    with pytest.raises(WebsiteFetchError, match="Timed out"):
        fetch_website_text("acme.com", session=session)


def test_fetch_website_text_connection_error():
    session = make_session(exc=requests.ConnectionError("dns failure"))
    with pytest.raises(WebsiteFetchError, match="Could not reach"):
        fetch_website_text("acme.com", session=session)


def test_fetch_website_text_http_error():
    session = make_session(status=404)
    with pytest.raises(WebsiteFetchError, match="HTTP 404"):
        fetch_website_text("acme.com", session=session)


@pytest.mark.parametrize("text", ["", "<script>x()</script>", "   "])
def test_fetch_website_text_empty_content(text):
    session = make_session(text=text)
    with pytest.raises(WebsiteFetchError, match="No readable content"):
        fetch_website_text("acme.com", session=session)


# --------------------------------------------------------------------------- #
# analyze_company
# --------------------------------------------------------------------------- #
def test_analyze_company_success():
    client = make_client(json.dumps(VALID_PAYLOAD))
    result = analyze_company("acme.com", "Acme widgets", client=client, sender_context="We do CPQ")

    assert isinstance(result, EnrichmentResult)
    assert result.domain == "acme.com"
    assert len(result.emails) == 3
    assert len(result.pain_points) == 2
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert "We do CPQ" in kwargs["messages"][1]["content"]


@pytest.mark.parametrize("text", ["", "   "])
def test_analyze_company_rejects_empty_text(text):
    client = make_client(json.dumps(VALID_PAYLOAD))
    with pytest.raises(ValueError):
        analyze_company("acme.com", text, client=client)
    client.chat.completions.create.assert_not_called()


def test_analyze_company_api_error():
    client = make_client(exc=OpenAIError("rate limited"))
    with pytest.raises(LLMResponseError, match="request failed"):
        analyze_company("acme.com", "text", client=client)


@pytest.mark.parametrize("content", [None, ""])
def test_analyze_company_empty_response(content):
    with pytest.raises(LLMResponseError, match="empty"):
        analyze_company("acme.com", "text", client=make_client(content))


def test_analyze_company_no_choices():
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[])
    with pytest.raises(LLMResponseError, match="unexpected response shape"):
        analyze_company("acme.com", "text", client=client)


def test_analyze_company_invalid_json():
    with pytest.raises(LLMResponseError, match="not valid JSON"):
        analyze_company("acme.com", "text", client=make_client("not json {"))


def test_analyze_company_json_not_object():
    with pytest.raises(LLMResponseError, match="not an object"):
        analyze_company("acme.com", "text", client=make_client("[1, 2, 3]"))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["emails"].pop(),                      # only 2 emails
        lambda p: p["emails"].append(p["emails"][0]),     # 4 emails
        lambda p: p.update(pain_points=[]),               # no pain points
        lambda p: p.pop("company_summary"),               # missing field
        lambda p: p["emails"][0].update(subject=""),      # empty subject
    ],
)
def test_analyze_company_schema_violations(mutate):
    payload = json.loads(json.dumps(VALID_PAYLOAD))
    mutate(payload)
    with pytest.raises(LLMResponseError, match="failed validation"):
        analyze_company("acme.com", "text", client=make_client(json.dumps(payload)))


# --------------------------------------------------------------------------- #
# enrich_lead / get_client / main CLI
# --------------------------------------------------------------------------- #
def test_enrich_lead_end_to_end_with_mocks():
    result = enrich_lead(
        "https://www.Acme.com/",
        client=make_client(json.dumps(VALID_PAYLOAD)),
        session=make_session(),
    )
    assert result.domain == "acme.com"
    assert len(result.emails) == 3


def test_enrich_lead_invalid_domain_skips_network():
    session, client = make_session(), make_client(json.dumps(VALID_PAYLOAD))
    with pytest.raises(InvalidDomainError):
        enrich_lead("not a domain", client=client, session=session)
    session.get.assert_not_called()
    client.chat.completions.create.assert_not_called()


def test_get_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        main.get_client()


def test_main_invalid_domain_exits_nonzero(capsys):
    assert main.main(["not a domain"]) == 1
    assert "Invalid domain" in capsys.readouterr().err


def test_main_missing_api_key_exits_nonzero(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "fetch_website_text", lambda d, session=None: "text")
    assert main.main(["acme.com"]) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_main_success_prints_json(monkeypatch, capsys):
    result = EnrichmentResult(domain="acme.com", **VALID_PAYLOAD)
    monkeypatch.setattr(main, "enrich_lead", lambda *a, **k: result)
    assert main.main(["acme.com", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["domain"] == "acme.com"
    assert len(out["emails"]) == 3


def test_main_success_prints_text(monkeypatch, capsys):
    result = EnrichmentResult(domain="acme.com", **VALID_PAYLOAD)
    monkeypatch.setattr(main, "enrich_lead", lambda *a, **k: result)
    assert main.main(["acme.com"]) == 0
    out = capsys.readouterr().out
    assert "Email draft 3" in out
    assert "Supply chain delays" in out
