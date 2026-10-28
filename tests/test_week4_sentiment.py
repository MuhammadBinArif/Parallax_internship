from src.week4.sentiment_analysis import VADERSentimentAnalyzer, TextBlobSentimentAnalyzer


def test_vader_positive():
    analyzer = VADERSentimentAnalyzer()
    result = analyzer.analyze("This is wonderful and amazing!")
    assert result.label == "positive"
    assert result.score > 0.05


def test_vader_negative():
    analyzer = VADERSentimentAnalyzer()
    result = analyzer.analyze("This is terrible and horrible.")
    assert result.label == "negative"
    assert result.score < -0.05


def test_vader_neutral():
    analyzer = VADERSentimentAnalyzer()
    result = analyzer.analyze("The document was filed on Tuesday.")
    assert result.label == "neutral"


def test_vader_empty():
    analyzer = VADERSentimentAnalyzer()
    result = analyzer.analyze("")
    assert result.label == "neutral"
    assert result.score == 0.0


def test_vader_batch():
    analyzer = VADERSentimentAnalyzer()
    texts = ["Great!", "Awful.", "Okay."]
    results = analyzer.analyze_batch(texts)
    assert len(results) == 3
    assert results[0].label == "positive"
    assert results[1].label == "negative"
