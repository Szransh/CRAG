from typing import List, TypedDict
from pydantic import BaseModel, Field
import os
import re
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END

# ============================================================
# ENV
# ============================================================
load_dotenv()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
if LLM_PROVIDER == "groq":
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("Missing GROQ_API_KEY")
elif LLM_PROVIDER == "openai":
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Missing OPENAI_API_KEY")

if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("Missing TAVILY_API_KEY")

print(f"\n🔹 Provider: {LLM_PROVIDER.upper()}")

# ============================================================
# LOAD DATA
# ============================================================
print("\n📂 Loading PDFs...")
docs = []
for root, _, files in os.walk("./documents"):
    for file in files:
        if file.lower().endswith(".pdf"):
            path = os.path.join(root, file)
            print(" Loading:", path)
            docs.extend(PyPDFLoader(path).load())

print("📄 Pages loaded:", len(docs))

chunks = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150
).split_documents(docs)

for d in chunks:
    d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

print("📦 Chunks created:", len(chunks))

# ============================================================
# EMBEDDINGS + VECTOR STORE
# ============================================================
print("\n🧠 Building vector store...")
if LLM_PROVIDER == "openai":
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
else:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 8})

# ============================================================
# LLM
# ============================================================
if LLM_PROVIDER == "groq":
    # Fixed: Added model keyword
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
else:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================
# STATE
# ============================================================
class State(TypedDict):
    question: str
    docs: List[Document]
    refined_context: str
    answer: str
    verdict: str    # Added to state
    web_query: str  # Added to state
    reason: str     # Added to state

# ============================================================
# RETRIEVE NODE
# ============================================================
def retrieve_node(state: State) -> State:
    print("\n➡️ RETRIEVE NODE")
    docs = retriever.invoke(state["question"])
    print(f"📚 Retrieved docs: {len(docs)}")
    return {"docs": docs}

# ============================================================
# SUPER GRADER (ULTRA FAST CORE)
# ============================================================
class SuperGrade(BaseModel):
    verdict: str = Field(description="CORRECT, INCORRECT, or AMBIGUOUS")
    kept_sentences: List[str]
    web_query: str

super_parser = PydanticOutputParser(pydantic_object=SuperGrade)
fmt = super_parser.get_format_instructions().replace("{", "{{").replace("}", "}}")

super_prompt = ChatPromptTemplate.from_messages([
    ("system", f"""
You are a corrective RAG grader.
Tasks:
1. Evaluate relevance of chunks.
2. Decide verdict = CORRECT / INCORRECT / AMBIGUOUS.
3. Extract only sentences useful for answering.
4. If info missing, produce web search query.
{fmt}
"""),
    ("human", "Question:{question}\n\nChunks:\n{chunks}")
])

tavily = TavilySearchResults(max_results=5)

def super_grade_node(state: State) -> State:
    print("\n➡️ SUPER GRADER NODE (ULTRA FAST)")
    docs = state["docs"]
    joined = "\n\n".join(
        [f"[{i}] {d.page_content[:500]}" for i, d in enumerate(docs)]
    )

    print("📦 Sending evaluation + refinement in ONE LLM call...")
    chain = super_prompt | llm

    # Default values for state
    current_verdict = "INCORRECT"
    current_web_query = ""
    current_reason = "Parsing error or no context found"

    try:
        raw = chain.invoke({
            "question": state["question"],
            "chunks": joined
        }).content

        out = super_parser.parse(raw)
        current_verdict = out.verdict
        current_web_query = out.web_query
        current_reason = "SuperGrade analysis complete"

        print("📊 Verdict:", out.verdict)
        context = "\n".join(out.kept_sentences)

        # ---------- PRODUCTION CORRECTIVE LOGIC ----------
        if out.verdict != "CORRECT" and out.web_query:
            print("\n🌐 Performing corrective web search...")
            results = tavily.invoke({"query": out.web_query})
            web_docs = []
            for r in results or []:
                txt = f"TITLE:{r.get('title','')}\nURL:{r.get('url','')}\nCONTENT:\n{r.get('content','')}"
                web_docs.append(txt)
            context += "\n\n" + "\n\n".join(web_docs)

        if not context.strip():
            context = "\n\n".join([d.page_content for d in docs])

    except Exception as e:
        print(f"⚠️ Super-Grader Parser Error: {e}")
        context = "\n\n".join([d.page_content for d in docs])

    return {
        "refined_context": context,
        "verdict": current_verdict,
        "web_query": current_web_query,
        "reason": current_reason
    }

# ============================================================
# GENERATION NODE (SECOND LLM CALL)
# ============================================================
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY using the provided context. If information is missing, say you don't know."),
    ("human", "Question:{question}\n\nContext:{context}")
])

def generate(state: State) -> State:
    print("\n➡️ GENERATION NODE")
    chain = answer_prompt | llm
    out = chain.invoke({
        "question": state["question"],
        "context": state["refined_context"]
    })
    return {"answer": out.content}

# ============================================================
# GRAPH
# ============================================================
g = StateGraph(State)
g.add_node("retrieve", retrieve_node)
g.add_node("super_grade", super_grade_node)
g.add_node("generate", generate)
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "super_grade")
g.add_edge("super_grade", "generate")
g.add_edge("generate", END)
app = g.compile()

# ============================================================
# MAIN
# ============================================================
# Fixed: Added double underscores
if __name__ == "__main__":
    q = input("\nEnter your question: ")

    initial_state = {
        "question": q,
        "docs": [],
        "refined_context": "",
        "answer": "",
        "verdict": "",
        "web_query": "",
        "reason": ""
    }

    print("\n🚀 Running PRODUCTION ULTRA FAST CRAG v2...\n")

    start = time.time()
    res = app.invoke(initial_state)
    end = time.time()

    print("\n==============================")
    print("VERDICT:", res.get("verdict"))
    print("REASON:", res.get("reason"))
    print("WEB_QUERY:", res.get("web_query"))
    print("\nOUTPUT:\n", res.get("answer"))
    print(f"\n⏱ Total Time: {end - start:.2f} seconds")
    print("==============================")