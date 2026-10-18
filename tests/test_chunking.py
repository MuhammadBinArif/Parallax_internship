from src.chunking import chunk_text, clean_text


def test_clean_text_none():
    assert clean_text(None) == ""


def test_clean_text_collapses_whitespace():
    assert clean_text("a  b\n\n\n\nc") == "a b\n\nc"


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text(None) == []


def test_short_text_returns_one_chunk():
    chunks = chunk_text("Hello world.", chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."


def test_long_text_respects_chunk_size():
    text = " ".join(f"sentence {i}." for i in range(1000))
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
    assert chunks
    assert all(len(c.text) <= 200 for c in chunks)


def test_hard_split_when_no_separators():
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=0)
    assert len(chunks) >= 10
    assert all(len(c.text) <= 100 for c in chunks)


def test_unicode_is_preserved():
    chunks = chunk_text("héllo wörld ✓", chunk_size=100)
    assert chunks[0].text == "héllo wörld ✓"


def test_metadata_includes_source():
    chunks = chunk_text("abc", source="doc.txt")
    assert chunks[0].metadata["source"] == "doc.txt"


def test_chunk_ids_are_deterministic():
    a = chunk_text("same text", source="doc.txt")
    b = chunk_text("same text", source="doc.txt")
    assert a[0].id == b[0].id
