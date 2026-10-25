from src.week3.validators import (
    extract_citations,
    normalize_category,
    parse_json_object,
    sentence_support,
)


def test_parse_json_object_plain():
    text = '{"category": "answered", "answer": "Yes.", "citations": ["doc-1"]}'
    parsed = parse_json_object(text)

    assert parsed["category"] == "answered"
    assert parsed["citations"] == ["doc-1"]


def test_parse_json_object_markdown():
    text = """```json
{
  "category": "out_of_domain",
  "answer": "",
  "citations": []
}
```"""

    parsed = parse_json_object(text)
    assert parsed["category"] == "out_of_domain"


def test_extract_citations():
    text = "Supported [doc-1]. Also [doc-2, doc-3]."
    assert extract_citations(text) == ["doc-1", "doc-2", "doc-3"]


def test_normalize_category():
    assert normalize_category("answered") == "answered"
    assert normalize_category("out-of-domain") == "out_of_domain"
    assert normalize_category("insufficient context") == "insufficient_context"
    assert normalize_category("weird") == "invalid"


def test_sentence_support():
    sentence = "Anarchism is a political philosophy that opposes hierarchical authority."
    evidence = "Anarchism is a political philosophy that opposes hierarchical authority."

    assert sentence_support(sentence, evidence) > 0.7
