"""
RAG with LangChain + Groq (ChatGroq) — single file
MentorX Academy · Module 3.1

What it does:
    Company documents -> split into chunks -> embed -> store in Chroma
    Question -> find the most similar chunks -> ChatGroq answers ONLY from them

Setup (once):
    pip install -U langchain-groq langchain-core langchain-text-splitters \
                   langchain-huggingface langchain-chroma sentence-transformers python-dotenv

    Create a .env file next to this script:
        GROQ_API_KEY=gsk_your_key_here
        GROQ_MODEL=llama-3.3-70b-versatile

Run:
    python rag_groq_langchain.py

Optional: put your own .txt / .md files in a "docs" folder next to this script —
they are added to the knowledge base automatically.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ═══════════════════════════════════════════════════════════════
# 1. CONFIG — keys come from .env, never from code
# ═══════════════════════════════════════════════════════════════
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 2               # how many chunks to retrieve per question
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
DOCS_DIR = Path(__file__).parent / "docs"

if not GROQ_API_KEY:
    sys.exit("❌ GROQ_API_KEY missing. Add it to your .env file.")

# ═══════════════════════════════════════════════════════════════
# 2. DOCUMENTS — built-in company policies (+ optional docs/ folder)
# ═══════════════════════════════════════════════════════════════
BUILT_IN_DOCS = [
    Document(
        page_content="""
        VENDOR ESCALATION POLICY — MentorX Operations. Effective: Jan 2026.
        Vendor misses delivery 1-2 days: Send automated warning email.
        Vendor misses delivery 3-5 days: Escalate to Procurement Manager.
        Vendor misses delivery more than 5 days: Invoke penalty clause.
        Penalty clause charges 2% of invoice value per day of delay.
        Repeat offenders (3+ incidents in 6 months) placed on Vendor Watch List.
        """,
        metadata={"source": "vendor_policy.pdf"},
    ),
    Document(
        page_content="""
        OVERTIME APPROVAL POLICY — MentorX Operations. Effective: Mar 2026.
        Overtime up to 20 hours per month: No pre-approval needed.
        Overtime 20-40 hours: Requires Warehouse Manager approval.
        Overtime above 40 hours: Requires VP Operations approval.
        Overtime rate is 1.5x the standard hourly rate.
        Monthly overtime budget cap is $30,000 for the main warehouse.
        If projected above $25,000: notify CFO by the 15th of that month.
        """,
        metadata={"source": "hr_policy.pdf"},
    ),
    Document(
        page_content="""
        RETURNS HANDLING POLICY — MentorX Operations. Effective: Feb 2026.
        Standard return window: 30 days from delivery.
        Return rate target: below 7% monthly.
        Return rate above 7%: root cause analysis within 5 business days.
        Return rate above 10%: immediate escalation to Quality Control Manager.
        Returns processed within 48 hours of receipt.
        Customer refund issued within 5 business days.
        """,
        metadata={"source": "returns_policy.pdf"},
    ),
    Document(
        page_content="""
        KPI TARGETS 2026 — MentorX Operations.
        Order Fulfillment Rate: above 95%.
        On-Time Delivery: above 92%.
        Inventory Accuracy: above 98%.
        Return Rate: below 7%.
        Overtime as percent of Labor Cost: below 15%.
        Cost per Unit Shipped: below $17.50.
        Warehouse Productivity: above 85 units per staff hour.
        Vendor On-Time Performance: above 90%.
        """,
        metadata={"source": "kpi_targets.pdf"},
    ),
]


def load_documents():
    """Built-in docs + any .txt/.md files found in ./docs."""
    docs = list(BUILT_IN_DOCS)
    if DOCS_DIR.is_dir():
        for path in sorted(DOCS_DIR.glob("*")):
            if path.suffix.lower() in (".txt", ".md"):
                text = path.read_text(encoding="utf-8", errors="ignore")
                docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


# ═══════════════════════════════════════════════════════════════
# 3. BUILD THE RAG PIPELINE
# ═══════════════════════════════════════════════════════════════
def build_retriever():
    docs = load_documents()
    print(f"📄 {len(docs)} documents loaded")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"✂️  {len(chunks)} chunks created")

    print("🧮 Embedding chunks (first run downloads ~90MB model)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)  # in-memory
    print("✅ Vector store ready")

    return vectordb.as_retriever(search_kwargs={"k": TOP_K})


def format_docs(docs):
    """Turn retrieved chunks into one context string, tagged with their source."""
    return "\n\n".join(f"[{d.metadata.get('source', '?')}]\n{d.page_content.strip()}" for d in docs)


RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are OPSAI. Answer ONLY from the context below.
If the answer is not in the context, say: "Not in the policy documents."
Always mention which document your answer comes from.

Context:
{context}

Question: {question}
Answer:"""
)


def build_rag_chain(retriever):
    """LCEL RAG chain: returns {'answer': str, 'sources': [Document]}."""
    llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL, temperature=0)

    answer_chain = (
        {
            "context": lambda x: format_docs(x["sources"]),
            "question": lambda x: x["question"],
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    # Step 1: retrieve chunks; Step 2: answer from them (keeping the chunks for citation)
    return RunnableParallel(
        sources=retriever, question=RunnablePassthrough()
    ).assign(answer=answer_chain)


def ask(chain, question):
    result = chain.invoke(question)
    sources = sorted({d.metadata.get("source", "?") for d in result["sources"]})
    return result["answer"], sources


# ═══════════════════════════════════════════════════════════════
# 4. CLI LOOP
# ═══════════════════════════════════════════════════════════════
SAMPLE_QUESTIONS = [
    "Our return rate in June was 8.9%. What does policy say?",
    "A vendor is 6 days late on delivery. What penalty applies?",
    "What is the overtime budget cap and who approves above 40 hours?",
]


def main():
    print("=" * 60)
    print(f"📚 OPSAI RAG · LangChain + Groq · Model: {MODEL}")
    print("=" * 60)

    retriever = build_retriever()
    chain = build_rag_chain(retriever)

    print("\n💡 Try:")
    for q in SAMPLE_QUESTIONS:
        print(f"   • {q}")
    print("\nType /exit to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            break

        if not question:
            continue
        if question.lower() in ("/exit", "/quit", "exit", "quit"):
            print("👋 Bye!")
            break

        try:
            answer, sources = ask(chain, question)
            print(f"\n🤖 {answer}")
            print(f"📄 Retrieved from: {', '.join(sources)}\n")
        except Exception as e:
            print(f"\n⚠️ Error: {e}\n")


if __name__ == "__main__":
    main()
