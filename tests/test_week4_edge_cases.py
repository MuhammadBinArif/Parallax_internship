from src.week4.edge_cases import (
    handle_edge_case,
    is_mostly_jargon,
    is_too_short,
    preprocess_for_nlp,
)


def test_empty_text():
    assert handle_edge_case("") == "empty"
    assert handle_edge_case(None) == "empty"
    assert handle_edge_case("   ") == "empty"


def test_short_text():
    assert handle_edge_case("Hi") == "too_short"
    assert handle_edge_case("This is a longer text that has enough words to pass the minimum threshold for analysis.") is None


def test_jargon_detection():
    jargon = "API SDK JSON REST GraphQL OAuth JWT SSL TLS HTTP"
    assert is_mostly_jargon(jargon) is True


def test_normal_text_not_jargon():
    normal = "Anarchism is a political philosophy that opposes hierarchical authority."
    assert is_mostly_jargon(normal) is False


def test_preprocess_removes_urls():
    text = "Visit https://example.com for info"
    cleaned = preprocess_for_nlp(text)
    assert "https://" not in cleaned


def test_preprocess_handles_empty():
    assert preprocess_for_nlp("") == ""
    assert preprocess_for_nlp(None) == ""
