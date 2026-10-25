from __future__ import annotations

from typing import Any, Iterable


SYSTEM_PROMPT = """\
You are a careful retrieval-augmented assistant.

Your job is to answer user queries using ONLY the provided context.

Strict rules:
1. Do not use prior knowledge.
2. Do not speculate.
3. Do not invent facts.
4. Every factual statement in your answer must be supported by the context.
5. Cite the chunk IDs you used inline, for example: [chunk-id].
6. If the query is unrelated to the available documents, classify it as out_of_domain.
7. If the query is related but the context does not contain enough information, classify it as insufficient_context.
8. If the user asks for creative writing, such as a poem, haiku, sonnet, story, joke, song, lyrics, or fiction, classify it as out_of_domain unless the documents themselves are about that creative task.
9. If the user asks for general trivia unrelated to the documents, such as sports, weather, stock prices, cryptocurrency prices, capitals, celebrities, or current events not present in the context, classify it as out_of_domain.
10. Respond with valid JSON only. Do not include markdown fences.
"""


JSON_INSTRUCTION = """\
Return exactly one JSON object with this schema:

{
  "category": "answered" | "insufficient_context" | "out_of_domain",
  "answer": "string",
  "citations": ["chunk-id"]
}

Field rules:
- category = "answered" only if the context contains sufficient information.
- category = "insufficient_context" if the topic is in-domain but evidence is missing.
- category = "out_of_domain" if the query is unrelated to the provided documents.
- category = "out_of_domain" for creative writing requests such as poems, stories, jokes, songs, or lyrics when the documents do not cover that request.
- answer = concise natural-language answer. Use inline citations like [chunk-id].
- citations = list of chunk IDs actually used.
- If category is not "answered", set answer to an empty string and citations to [].
"""


STRICT_REVISION_INSTRUCTION = """\
STRICT REVISION REQUIRED.

Your previous answer was not fully supported by the retrieved context.

Rewrite the answer using ONLY facts that appear directly in the context.
Every sentence must include at least one citation.
Remove any claim that cannot be verified from the provided chunks.
Keep the answer short and precise.
"""


def xml_escape(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def format_context(chunks: Iterable[Any]) -> str:
    blocks = []

    for chunk in chunks:
        chunk_id = xml_escape(getattr(chunk, "id", "unknown"))
        chunk_text = xml_escape(getattr(chunk, "text", ""))
        blocks.append(
            f'<chunk id="{chunk_id}">\n{chunk_text}\n</chunk>'
        )

    return "\n".join(blocks)


def build_messages(
    query: str,
    chunks: Iterable[Any],
    extra_instruction: str | None = None,
) -> list[dict[str, str]]:
    context = format_context(chunks)
    safe_query = xml_escape(query)

    extra = ""
    if extra_instruction:
        extra = f"\n\nAdditional instruction:\n{extra_instruction}"

    user_prompt = f"""\
<context>
{context}
</context>

<query>
{safe_query}
</query>

{JSON_INSTRUCTION}{extra}
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
