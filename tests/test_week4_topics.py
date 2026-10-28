from src.week4.topic_modeling import LDATopicModeler


def test_lda_fit_and_predict():
    texts = [
        "Anarchism is a political philosophy.",
        "The stock market crashed today.",
        "Workers organized a general strike.",
        "The government passed new legislation.",
        "Markets responded to interest rate changes.",
    ] * 10  # Repeat for minimum samples

    modeler = LDATopicModeler(n_topics=2, max_features=100)
    modeler.fit(texts)

    results = modeler.predict(["Anarchism opposes the state."])
    assert len(results) == 1
    assert results[0].topic_id in [0, 1]
    assert results[0].probability >= 0.0


def test_lda_not_fitted_raises():
    import pytest
    modeler = LDATopicModeler(n_topics=2)
    with pytest.raises(RuntimeError):
        modeler.predict(["test"])
