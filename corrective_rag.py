from typing import List, TypedDict
from pydantic import BaseModel
import re
import os
import time

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langgraph.graph import StateGraph, START, END

# ============================================================
# ENV + CONFIG
# ============================================================

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

if LLM_PROVIDER == "groq":
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("Set GROQ_API_KEY environment variable.")
elif LLM_PROVIDER == "openai":
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY environment variable.")
else:
    raise ValueError("LLM_PROVIDER must be 'groq' or 'openai'")

print(f"🔹 Using provider: {LLM_PROVIDER.upper()}")

# ============================================================
# DATA LOADING
# ============================================================

print("🔹 Loading PDF documents recursively...")

docs = []
for root, _, files in os.walk("./documents"):
    for file in files:
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(root, file)
            print(f"   Loading: {pdf_path}")
            loaded = PyPDFLoader(pdf_path).load()
            docs.extend(loaded)

print(f"   Total PDFs loaded: {len(docs)} raw document pages")

print("🔹 Splitting into chunks...")
chunks = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150
).split_documents(docs)

print(f"   Created {len(chunks)} chunks")

for d in chunks:
    d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

# ============================================================
# EMBEDDINGS
# ============================================================

if LLM_PROVIDER == "openai":
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
else:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

print("🔹 Building FAISS index...")
vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8}
)

# ============================================================
# LLM INITIALIZATION
# ============================================================

if LLM_PROVIDER == "groq":
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
else:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

UPPER_TH = 0.7
LOWER_TH = 0.3

# ============================================================
# STATE
# ============================================================

class State(TypedDict):
    question: str
    docs: List[Document]
    good_docs: List[Document]
    verdict: str
    reason: str
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    web_query: str
    web_docs: List[Document]
    answer: str

# ============================================================
# RETRIEVE
# ============================================================

def retrieve_node(state: State) -> State:
    print("\n➡️  RETRIEVE NODE")
    docs = retriever.invoke(state["question"])
    print(f"   Retrieved {len(docs)} documents")
    return {"docs": docs}

# ============================================================
# DOC EVALUATION
# ============================================================

class DocEvalScore(BaseModel):
    score: float
    reason: str

doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Return relevance score in [0.0,1.0] and short reason. Output JSON Format",
        ),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ]
)

doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)

def eval_each_doc_node(state: State) -> State:
    print("\n➡️  EVALUATION NODE")
    scores: List[float] = []
    good: List[Document] = []

    for i, d in enumerate(state["docs"]):
        print(f"   Evaluating doc {i+1}/{len(state['docs'])}")
        out = doc_eval_chain.invoke({"question": state["question"], "chunk": d.page_content})
        scores.append(out.score)

        if out.score > LOWER_TH:
            good.append(d)

    if any(s > UPPER_TH for s in scores):
        return {"good_docs": good, "verdict": "CORRECT", "reason": "High scoring chunk found."}

    if len(scores) > 0 and all(s < LOWER_TH for s in scores):
        return {"good_docs": [], "verdict": "INCORRECT", "reason": "All chunks low score."}

    return {"good_docs": good, "verdict": "AMBIGUOUS", "reason": "Mixed scores."}

# ============================================================
# REFINEMENT
# ============================================================

class KeepOrDrop(BaseModel):
    keep: bool

filter_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Return keep=true only if sentence helps answer question. JSON only."),
        ("human", "Question: {question}\n\nSentence:\n{sentence}"),
    ]
)

filter_chain = filter_prompt | llm.with_structured_output(KeepOrDrop)

def decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

def refine(state: State) -> State:
    print("\n➡️  REFINEMENT NODE")

    if state["verdict"] == "CORRECT":
        docs_to_use = state["good_docs"]
    elif state["verdict"] == "INCORRECT":
        docs_to_use = state["web_docs"]
    else:
        docs_to_use = state["good_docs"] + state["web_docs"]

    context = "\n\n".join(d.page_content for d in docs_to_use).strip()
    strips = decompose_to_sentences(context)

    kept = []
    for s in strips:
        if filter_chain.invoke({"question": state["question"], "sentence": s}).keep:
            kept.append(s)

    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": "\n".join(kept).strip(),
    }

# ============================================================
# WEB SEARCH
# ============================================================

class WebQuery(BaseModel):
    query: str

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Rewrite into web search keywords. JSON only."),
        ("human", "Question: {question}"),
    ]
)

rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)

def rewrite_query_node(state: State) -> State:
    out = rewrite_chain.invoke({"question": state["question"]})
    return {"web_query": out.query}

tavily = TavilySearchResults(max_results=5)

def web_search_node(state: State) -> State:
    results = tavily.invoke({"query": state.get("web_query") or state["question"]})
    web_docs = []

    for r in results or []:
        text = f"TITLE: {r.get('title','')}\nURL: {r.get('url','')}\nCONTENT:\n{r.get('content','')}"
        web_docs.append(Document(page_content=text))

    return {"web_docs": web_docs}

# ============================================================
# GENERATE
# ============================================================

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer ONLY using context."),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)

def generate(state: State) -> State:
    out = (answer_prompt | llm).invoke(
        {"question": state["question"], "context": state["refined_context"]}
    )
    return {"answer": out.content}

# ============================================================
# ROUTING
# ============================================================

def route_after_eval(state: State) -> str:
    return "refine" if state["verdict"] == "CORRECT" else "rewrite_query"

# ============================================================
# GRAPH
# ============================================================

g = StateGraph(State)

g.add_node("retrieve", retrieve_node)
g.add_node("eval_each_doc", eval_each_doc_node)
g.add_node("rewrite_query", rewrite_query_node)
g.add_node("web_search", web_search_node)
g.add_node("refine", refine)
g.add_node("generate", generate)

g.add_edge(START, "retrieve")
g.add_edge("retrieve", "eval_each_doc")

g.add_conditional_edges(
    "eval_each_doc",
    route_after_eval,
    {"refine": "refine", "rewrite_query": "rewrite_query"},
)

g.add_edge("rewrite_query", "web_search")
g.add_edge("web_search", "refine")
g.add_edge("refine", "generate")
g.add_edge("generate", END)

app = g.compile()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    question = input("\nEnter your question: ")

    initial_state = {
        "question": question,
        "docs": [],
        "good_docs": [],
        "verdict": "",
        "reason": "",
        "strips": [],
        "kept_strips": [],
        "refined_context": "",
        "web_query": "",
        "web_docs": [],
        "answer": "",
    }

    print("\n🚀 Running CRAG pipeline...\n")
    start = time.time()

    res = app.invoke(initial_state)

    end = time.time()

