from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import re

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
SIMILARITY_THRESHOLD = 0.4
CHROMA_PATH = "./chroma_db"
TOP_K = 3

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name="pdf_chunks",
    metadata={"hnsw:space": "cosine"},
)


def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


SECTION_KEYWORDS = {
    "abstract", "introduction", "background", "literature review",
    "related work", "methodology", "methods", "system design",
    "implementation", "results", "discussion", "evaluation",
    "conclusion", "future work", "references", "acknowledgements",
    "acknowledgments", "objectives", "scope", "overview",
    "problem statement", "proposed system", "architecture",
    "testing", "analysis", "summary",
    # common sub-section headings
    "functional scope", "technical scope", "limitations",
    "technology stack", "expected outcomes", "expected outcome",
    "phases", "phase", "tools used", "tools and technologies",
    "features", "requirements", "system requirements",
    "hardware requirements", "software requirements",
}


def _is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 100:
        return False
    words = line.split()
    # Numbered section: "1.", "1.1", "2.3.1 Title"
    if re.match(r'^\d+(\.\d+)*\.?\s+\w', line):
        return True
    # Known section keyword at start of a short line
    lower = line.lower().rstrip(".:")
    for kw in SECTION_KEYWORDS:
        if lower == kw or lower.startswith(kw + " "):
            if len(words) <= 7:
                return True
    # ALL CAPS short line (common heading style in PDFs)
    if line.isupper() and 1 <= len(words) <= 8:
        return True
    return False


def _extract_qa_pairs(lines):
    """Split body lines into one chunk per Q&A pair when Q patterns exist."""
    pairs = []
    current = []
    found_q = False
    for line in lines:
        if re.match(r'^Q\d*[\.:]?\s', line, re.IGNORECASE):
            if current:
                pairs.append("\n".join(current))
            current = [line]
            found_q = True
        else:
            current.append(line)
    if current:
        pairs.append("\n".join(current))
    return pairs if found_q else []


def split_text(text, chunk_size=250, overlap=50):
    """Section-aware chunking:
    - FAQ/Q&A sections  → one chunk per Q&A pair (precise retrieval)
    - Regular sections  → sliding window within that section only
    Each chunk is prefixed with its section heading so embeddings stay section-scoped.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Group lines into (heading, body_lines) sections
    sections = []
    current_heading = ""
    current_body: list[str] = []
    for line in lines:
        if _is_heading(line):
            if current_body:
                sections.append((current_heading, current_body))
            current_heading = line
            current_body = []
        else:
            current_body.append(line)
    if current_body:
        sections.append((current_heading, current_body))

    chunks = []
    for heading, body_lines in sections:
        prefix = f"{heading}\n" if heading else ""

        # FAQ/Q&A sections: one chunk per Q&A pair
        qa_pairs = _extract_qa_pairs(body_lines)
        if qa_pairs:
            for qa in qa_pairs:
                chunks.append(prefix + qa)
            continue

        # Regular text: sliding window scoped to this section
        body = " ".join(body_lines)
        words = body.split()
        if not words:
            continue
        if len(words) <= chunk_size:
            chunks.append(prefix + body)
        else:
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunks.append(prefix + " ".join(words[start:end]))
                if end == len(words):
                    break
                start += chunk_size - overlap

    return [c for c in chunks if c.strip()]


def create_vector_store(text, pdf_name):
    chunks = split_text(text)
    print(f"[VectorDB] Split into {len(chunks)} chunks for '{pdf_name}'")

    embeddings = embedding_model.encode(
        chunks, normalize_embeddings=True, show_progress_bar=False
    ).tolist()

    existing = collection.get(where={"pdf_name": pdf_name})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    ids = [f"{pdf_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "pdf_name": pdf_name,
            "chunk_index": i,
            "section": chunks[i].split("\n")[0][:60],
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    print(f"[VectorDB] Stored {len(chunks)} chunks for '{pdf_name}'")
    return len(chunks)


_STOPWORDS = {
    "what", "is", "the", "in", "my", "of", "a", "an", "are", "was", "were",
    "tell", "me", "about", "who", "when", "where", "how", "which", "find",
    "give", "show", "list", "describe", "explain", "document", "proposal",
    "pdf", "file", "paper", "this", "that", "these", "those", "and", "or",
    "for", "to", "from", "with", "by", "at", "on", "it", "its", "can", "do",
    "does", "did", "will", "would", "could", "should", "has", "have", "had",
    "i", "you", "he", "she", "we", "they", "please", "any", "all", "some",
    "no", "not", "also", "according", "mentioned", "written",
}


def _extract_key_terms(question: str) -> list[str]:
    """Pull out meaningful search tokens, drop stopwords and short words."""
    words = re.findall(r'\b[a-zA-Z0-9]+\b', question.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


# Maps common question intents to the vocabulary the document actually uses.
# "author" → adds "submitted by student name" so the embedding matches the BY section.
_INTENT_EXPANSIONS: dict[str, str] = {
    "author":       "submitted by student name written",
    "wrote":        "submitted by author student name",
    "made":         "submitted by author student",
    "college":      "vidyapeeth university institution submitted centre online learning",
    "university":   "vidyapeeth college institution centre online learning",
    "institution":  "vidyapeeth university college submitted",
    "submitted":    "by author student submitted",
    "supervisor":   "guide mentor internal supervisor professor",
    "guide":        "supervisor mentor internal professor",
    "mentor":       "supervisor guide internal professor",
    "advisor":      "supervisor guide internal professor",
    "title":        "project title chatbot development ai powered",
    "topic":        "project title specialization subject",
    "subject":      "specialization branch stream topic project",
    "specialization": "artificial intelligence machine learning branch stream",
    "stream":       "specialization artificial intelligence machine learning",
    "branch":       "specialization stream artificial intelligence",
    "student":      "submitted by author name erp prn",
    "name":         "student author submitted suraj erp",
    "year":         "2024 2025 2026 batch duration",
    "duration":     "year batch 2024 2026",
    "batch":        "year duration 2024 2026",
}


def _expand_query(question: str) -> str:
    """Append synonym terms for any recognised intent words in the question."""
    q_lower = question.lower()
    extras: list[str] = []
    for key, expansion in _INTENT_EXPANSIONS.items():
        if re.search(rf'\b{re.escape(key)}\b', q_lower):
            extras.append(expansion)
    return (question + " " + " ".join(extras)).strip() if extras else question


def search_similar_chunks(question, top_k=TOP_K):
    total = collection.count()
    if total == 0:
        return []

    print("\n--- Retrieval Results ---")
    question_lower = question.lower().strip()

    # Expand query with intent synonyms so "author" finds "BY", "college" finds "Vidyapeeth", etc.
    expanded_question = _expand_query(question)
    expanded_lower = expanded_question.lower()
    if expanded_question != question:
        print(f"  [expand]   '{question}' → '{expanded_question[:80]}'")

    # Load all chunks once — used by heading match and keyword search
    all_data = collection.get(include=["documents", "metadatas"])
    all_ids = all_data["ids"]
    all_docs = all_data["documents"]
    all_metas = all_data["metadatas"]

    output = []
    seen_ids: set[str] = set()

    # --- Stage 1: Heading match (original + expanded) ---
    for chunk_id, doc, meta in zip(all_ids, all_docs, all_metas):
        section = meta.get("section", "").lower().strip()
        if section and (
            question_lower in section or section in question_lower
            or expanded_lower in section or section in expanded_lower
        ):
            key = f"{meta['pdf_name']}_{meta['chunk_index']}"
            if key not in seen_ids:
                seen_ids.add(key)
                output.append({"chunk": doc, "similarity": 0.95, "pdf_name": meta["pdf_name"], "match_type": "heading"})
                print(f"  [heading]  {meta['pdf_name']} | {meta.get('section','')[:40]}")

    # --- Stage 2: Keyword match (expanded terms) ---
    if not output:
        # Use expanded query terms so "author" → also tries "submitted", "student", etc.
        key_terms = list(set(_extract_key_terms(question_lower) + _extract_key_terms(expanded_lower)))
        if key_terms:
            scored = []
            for chunk_id, doc, meta in zip(all_ids, all_docs, all_metas):
                key = f"{meta['pdf_name']}_{meta['chunk_index']}"
                if key in seen_ids:
                    continue
                doc_lower = doc.lower()
                matched = [t for t in key_terms if t in doc_lower]
                if matched:
                    ratio = len(matched) / len(key_terms)
                    sim = round(0.50 + ratio * 0.40, 4)
                    scored.append({"chunk": doc, "similarity": sim,
                                   "pdf_name": meta["pdf_name"], "match_type": "keyword", "_key": key})

            scored.sort(key=lambda x: x["similarity"], reverse=True)
            for item in scored[:top_k]:
                seen_ids.add(item.pop("_key"))
                output.append(item)
                print(f"  [keyword]  {item['pdf_name']} | sim: {item['similarity']:.4f} | terms: {key_terms[:6]}")

    # --- Stage 3: Semantic search (expanded query for better intent matching) ---
    prefixed_query = f"Represent this sentence for searching relevant passages: {expanded_question}"
    query_embedding = embedding_model.encode(
        [prefixed_query], normalize_embeddings=True
    ).tolist()

    sem_results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, total),
        include=["documents", "distances", "metadatas"],
    )

    for chunk, dist, meta in zip(
        sem_results["documents"][0], sem_results["distances"][0], sem_results["metadatas"][0]
    ):
        key = f"{meta['pdf_name']}_{meta['chunk_index']}"
        if key in seen_ids:
            continue
        similarity = round(1.0 - dist, 4)
        if similarity >= SIMILARITY_THRESHOLD:
            output.append({"chunk": chunk, "similarity": similarity, "pdf_name": meta["pdf_name"], "match_type": "semantic"})
            seen_ids.add(key)
            print(f"  [semantic] {meta['pdf_name']} | {meta.get('section','')[:40]} | sim: {similarity:.4f}")

    output = sorted(output, key=lambda x: x["similarity"], reverse=True)[:top_k]
    print(f"  Returning {len(output)} chunk(s)")
    print("-------------------------\n")

    return output


def list_indexed_pdfs():
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    return list({m["pdf_name"] for m in all_meta})
