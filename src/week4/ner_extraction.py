from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NERResult:
    entities: list[dict] = field(default_factory=list)
    entity_count: int = 0
    has_person: bool = False
    has_org: bool = False
    has_location: bool = False


class SpacyNERExtractor:
    """
    Named Entity Recognition using spaCy.
    Uses en_core_web_sm (already installed from Week 1).
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        import spacy
        self.nlp = spacy.load(model_name, disable=["parser", "lemmatizer"])
        self.nlp.max_length = 100000  # Handle long docs

    def extract(self, text: str, max_entities: int = 20) -> NERResult:
        if not text or not text.strip():
            return NERResult()

        truncated = text[:10000]
        doc = self.nlp(truncated)

        entities = []
        has_person = False
        has_org = False
        has_location = False

        for ent in doc.ents[:max_entities]:
            entity = {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            entities.append(entity)

            if ent.label_ == "PERSON":
                has_person = True
            elif ent.label_ in {"ORG", "NORP"}:
                has_org = True
            elif ent.label_ in {"GPE", "LOC", "FAC"}:
                has_location = True

        return NERResult(
            entities=entities,
            entity_count=len(entities),
            has_person=has_person,
            has_org=has_org,
            has_location=has_location,
        )

    def extract_batch(self, texts: list[str], max_entities: int = 20) -> list[NERResult]:
        return [self.extract(t, max_entities) for t in texts]
