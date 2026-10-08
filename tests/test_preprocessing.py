from src.preprocessing import (
    clean_document,
    clean_text,
    clean_whitespace,
    extract_text,
    is_mostly_english,
    iter_clean_documents,
    normalize_unicode,
    strip_html,
)


ENGLISH_LONG = (
    "The quick brown fox jumps over the lazy dog. "
    "The sun was shining brightly, and the birds were singing in the trees near the river. "
)

FRENCH_LONG = (
    "Le renard brun rapide saute par dessus le chien paresseux. "
    "Pendant que le soleil brillait, les oiseaux chantaient dans les arbres près de la rivière. "
)


def test_strip_html_removes_tags():
    out = strip_html("<p>Hello <b>world</b></p>")
    assert "Hello" in out
    assert "world" in out
    assert "<p>" not in out
    assert "<b>" not in out


def test_normalize_unicode_fullwidth_characters():
    assert normalize_unicode("\uff21\uff22\uff23") == "ABC"


def test_clean_whitespace_collapses_spaces_and_newlines():
    assert clean_whitespace("a  b\n\n\n\nc") == "a b\n\nc"
    assert clean_whitespace("  a \n b  ") == "a\nb"


def test_clean_text_strips_html_urls_and_emails():
    text = "<p>Visit https://example.com or email test@example.com for info.</p>"
    cleaned = clean_text(text)

    assert "https://" not in cleaned
    assert "test@" not in cleaned
    assert "Visit" in cleaned
    assert "info" in cleaned


def test_is_mostly_english_accepts_english():
    assert is_mostly_english(ENGLISH_LONG, min_tokens=10) is True


def test_is_mostly_english_rejects_french():
    assert is_mostly_english(FRENCH_LONG, min_tokens=10) is False


def test_extract_text_joins_selected_fields():
    doc = {
        "intro": "First part.",
        "life": "Second part.",
        "name": "This should not appear",
    }

    out = extract_text(doc, ("intro", "life"))
    assert out == "First part.\n\nSecond part."


def test_clean_document_returns_standard_schema():
    doc = {"text": ENGLISH_LONG}

    out = clean_document(
        doc,
        doc_id="x1",
        source="test",
        text_fields=("text",),
        min_chars=20,
        max_chars=10000,
        require_english=True,
    )

    assert out is not None
    assert out["id"] == "x1"
    assert out["source"] == "test"
    assert out["char_count"] == len(out["text"])
    assert out["word_count"] > 0
    assert out["language"] == "en_heuristic"


def test_clean_document_filters_short_document():
    out = clean_document(
        {"text": "Too short"},
        doc_id="short-1",
        source="test",
        text_fields=("text",),
        min_chars=100,
        max_chars=1000,
        require_english=True,
    )

    assert out is None


def test_iter_clean_documents_assigns_ids_and_skips_invalid():
    docs = [
        {"text": ENGLISH_LONG},
        {"text": "short"},
        "not-a-dict",
        {"text": ENGLISH_LONG},
    ]

    outs = list(
        iter_clean_documents(
            docs,
            source="unit",
            text_fields=("text",),
            min_chars=20,
            max_chars=1000,
            require_english=True,
        )
    )

    assert len(outs) == 2
    assert outs[0]["id"] == "unit-0"
    assert outs[1]["id"] == "unit-3"
