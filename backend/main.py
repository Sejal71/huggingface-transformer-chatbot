from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import re
import random
import uvicorn

from rag import extract_text_from_pdf, create_vector_store, search_similar_chunks, list_indexed_pdfs

GREETINGS = {
    "greet": {
        "patterns": [
            r"^(hi|hello|hey|howdy|hiya|sup|what'?s up|greetings|good\s*(morning|afternoon|evening|day))[!?.,]?$"
        ],
        "responses": [
            "Hello! How can I help you today? You can upload a PDF and ask me questions about it.",
            "Hey there! Ready to help. Upload a PDF document and I'll answer your questions about it.",
            "Hi! I'm your PDF assistant. Upload a document and ask away!",
        ],
    },
    "how_are_you": {
        "patterns": [
            r"how (are you|is it going|do you do|have you been|r u)[?!.,]?$",
            r"how'?s (it going|everything|life|things)[?!.,]?$",
            r"(what'?s up|wassup|how goes it)[?!.,]?$",
        ],
        "responses": [
            "I'm doing great, thanks for asking! Ready to help you explore your PDF documents.",
            "All good here! I'm here to answer questions about any PDF you upload.",
            "Doing well! Let me know what document you'd like to dive into.",
        ],
    },
    "capabilities": {
        "patterns": [
            r"what (can you do|are you capable of|do you do|are your capabilities|can you help with)[?!.,]?$",
            r"(what'?s your purpose|how do you work|tell me about yourself|who are you|what are you)[?!.,]?$",
            r"help$",
            r"(show me|list) (your )?(features|capabilities|functions)[?!.,]?$",
        ],
        "responses": [
            (
                "I'm a PDF Q&A assistant! Here's what I can do:\n\n"
                "Upload PDFs — Send me any PDF document to index.\n"
                "Answer Questions — Ask anything about the content of your uploaded PDFs.\n"
                "Multi-document — I can handle multiple PDFs at once and tell you which one the answer came from.\n\n"
                "Just upload a PDF and start asking questions!"
            ),
        ],
    },
    "thanks": {
        "patterns": [
            r"^(thanks|thank you|thank u|thx|ty|cheers)[!.,]?$",
            r"^(thanks|thank you) (so much|a lot|very much)[!.,]?$",
        ],
        "responses": [
            "You're welcome! Feel free to ask more questions.",
            "Happy to help! Let me know if you have more questions.",
            "Anytime! Ask me anything about your documents.",
        ],
    },
    "goodbye": {
        "patterns": [
            r"^(bye|goodbye|see you|see ya|later|cya|good night|good bye)[!.,]?$",
        ],
        "responses": [
            "Goodbye! Come back anytime you have documents to explore.",
            "See you later! Happy to help whenever you need.",
            "Bye! Feel free to return with more questions.",
        ],
    },
}


def get_casual_response(text: str):
    normalized = text.strip().lower()
    for category, data in GREETINGS.items():
        for pattern in data["patterns"]:
            if re.search(pattern, normalized):
                return random.choice(data["responses"])
    return None


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MIN_SIMILARITY = 0.55


def _clean_chunk(chunk: str) -> str:
    """Strip leading Q line from Q&A chunks; return section text as-is."""
    lines = chunk.strip().split("\n")
    result = []
    skipped_q = False
    for line in lines:
        if not skipped_q and re.match(r'^Q\d*[\.:]?\s', line, re.IGNORECASE):
            skipped_q = True
            continue
        if line.strip():
            result.append(line.strip())
    return "\n".join(result) if result else chunk


def _is_qa_chunk(chunk: str) -> bool:
    """True when the chunk is a single Q&A pair (not a document section)."""
    return bool(re.search(r'^Q\d*[\.:]?\s', chunk, re.IGNORECASE | re.MULTILINE))


_FACT_STOPWORDS = {
    "what", "is", "the", "in", "my", "of", "a", "an", "are", "tell", "me",
    "about", "who", "when", "where", "how", "which", "give", "show", "find",
    "document", "proposal", "pdf", "file", "for", "to", "from", "with",
    "it", "its", "this", "that", "and", "or", "no", "not", "was", "were",
}


def _focused_answer(chunk: str, question: str) -> str:
    """For keyword-matched chunks, return just the relevant snippet instead of the whole chunk."""
    terms = [
        w for w in re.findall(r'\b[a-zA-Z0-9]+\b', question.lower())
        if w not in _FACT_STOPWORDS and len(w) > 2
    ]
    if not terms:
        return chunk

    for term in terms:
        # Find "LABEL ... : " prefix
        m_label = re.search(rf'(?i)\b{re.escape(term)}\b[^:\n]{{0,15}}:\s*', chunk)
        if m_label:
            remaining = chunk[m_label.end():]
            value_words = []
            for w in remaining.split():
                if w == "•":
                    break
                # Stop at the next field label (word ending with ":" or ALL-CAPS short token)
                if value_words and (w.endswith(":") or (w.isupper() and len(w) > 1 and w.isalpha())):
                    break
                value_words.append(w)
                # Numeric IDs are single tokens — stop immediately after
                if re.match(r'^\d+$', w):
                    break
            if value_words:
                label = m_label.group(0).strip()
                return f"{label} {' '.join(value_words)}"

    # Fallback: return a small word-window starting at the first matched term
    words = chunk.split()
    words_lower = [w.lower() for w in words]
    for term in terms:
        for i, w in enumerate(words_lower):
            if term in w:
                end = min(len(words), i + 10)
                return " ".join(words[i:end])

    return chunk


@app.get("/")
def home():
    return {"message": "AI-Powered PDF Question Answering Backend is running"}


@app.get("/list-pdfs")
def list_pdfs():
    return {"pdfs": list_indexed_pdfs()}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    text = extract_text_from_pdf(file_path)
    total_chunks = create_vector_store(text, pdf_name=file.filename)
    return {
        "message": f"'{file.filename}' uploaded and indexed successfully",
        "chunks": total_chunks,
    }


@app.post("/ask-document")
async def ask_document(data: dict):
    question = data.get("question", "").strip()

    # Handle greetings and casual queries without touching the vector store
    casual = get_casual_response(question)
    if casual:
        return {"answer": casual, "similarity_scores": [], "sources": []}

    results = search_similar_chunks(question)

    if not results:
        return {
            "answer": "I couldn't find relevant information about this in the uploaded documents. Please try rephrasing your question.",
            "similarity_scores": [],
            "sources": [],
        }

    top_score = results[0]["similarity"]
    if top_score < MIN_SIMILARITY:
        return {
            "answer": (
                f"I couldn't find a confident match in your documents (best score: {top_score:.0%}). "
                "Try rephrasing or ask something more specific to the document content."
            ),
            "similarity_scores": [r["similarity"] for r in results],
            "sources": [],
        }

    top_result = results[0]
    top_chunk = top_result["chunk"]
    match_type = top_result.get("match_type", "semantic")

    if _is_qa_chunk(top_chunk):
        # Single Q&A pair — never merge sibling pairs
        answer = _clean_chunk(top_chunk)
    elif match_type == "keyword":
        # Specific fact lookup (PRN No., college name, etc.) — extract just the snippet
        answer = _focused_answer(top_chunk, question)
    else:
        # Section / semantic — only combine chunks from the same section heading
        top_heading = top_chunk.split("\n")[0].strip().lower()
        same_section = [
            r for r in results
            if r["chunk"].split("\n")[0].strip().lower() == top_heading
        ]
        answer = "\n\n".join(_clean_chunk(r["chunk"]) for r in same_section)

    return {
        "answer": answer,
        "similarity_scores": [r["similarity"] for r in results],
        "sources": [r["pdf_name"] for r in results],
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
