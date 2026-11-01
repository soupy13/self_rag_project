"""LangGraph Self-RAG workflow.
retrieve -> grade_documents -> check_answerability -> (generate | absence)
generate -> verify (dual-path, parallel) -> grade_answer -> (generate retry | END)"""
import json, operator, os, re, time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, List, TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
import store

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 1))
TOP_K = int(os.getenv("TOP_K", 6))
CTX_CHARS = int(os.getenv("CTX_CHARS", 1200))  # keeps each call small for free-tier token limits


def _llm(json_mode=False):
    name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    kw = {"model": name, "temperature": 0, "max_retries": 4}
    effort = os.getenv("REASONING_EFFORT", "low")
    m = None
    if "gpt-oss" in name and effort:
        try:  # newer langchain-groq takes it as an explicit parameter
            m = ChatGroq(reasoning_effort=effort, **kw)
        except Exception:
            m = None
    m = m or ChatGroq(**kw)
    return m.bind(response_format={"type": "json_object"}) if json_mode else m


def _json(prompt):
    """JSON call. API errors propagate (shown in the UI); only parsing is forgiving."""
    try:
        msg = _llm(True).invoke(prompt).content
    except Exception:
        msg = _llm().invoke(prompt + "\nRespond with a single JSON object only.").content
    for cand in (msg, (re.search(r"\{.*\}", msg, re.S) or [None])[0] if re.search(r"\{.*\}", msg, re.S) else None):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return {}


def _ctx(docs, limit=None):
    limit = limit or CTX_CHARS
    return "\n\n".join(f"[{i + 1}] ({d['source']}, {d['location']})\n{d['text'][:limit]}" for i, d in enumerate(docs))


class State(TypedDict, total=False):
    question: str
    session: str
    docs: list
    relevant: list
    answerable: bool
    missing: str
    answer: str
    verdict: dict
    feedback: str
    retries: int
    events: Annotated[List[dict], operator.add]


def _ev(node, t0, summary, **data):
    return {"node": node, "ms": int((time.time() - t0) * 1000), "summary": summary, "data": data}


def retrieve(s):
    t = time.time()
    docs = store.query(s["session"], s["question"], TOP_K)
    return {"docs": docs, "retries": 0, "events": [_ev("retrieve", t, f"Retrieved {len(docs)} chunks: " + "; ".join(sorted({d["source"] + " / " + d["location"] for d in docs[:3]})))]}


def grade_documents(s):
    t = time.time()
    docs = s["docs"]
    if not docs:
        return {"relevant": [], "events": [_ev("grade_documents", t, "No chunks to grade")]}
    r = _json(f"Question: {s['question']}\n\nChunks:\n{_ctx(docs, 400)}\n\nReturn JSON {{\"relevant\": [numbers of chunks "
              f"that could help answer the question]}}. Be inclusive: keep any chunk that may be useful.")
    lst = next((v for v in r.values() if isinstance(v, list)), [])
    ids = []
    for x in lst:
        try:
            i = int(x)
        except Exception:
            continue
        if 1 <= i <= len(docs) and i not in ids:
            ids.append(i)
    fallback = not ids
    if fallback:  # fail-open: never discard everything because the grader was strict or malformed
        ids = list(range(1, min(4, len(docs)) + 1))
    rel = [docs[i - 1] for i in ids[:3]]
    note = " (grader kept none; using top chunks)" if fallback else ""
    return {"relevant": rel, "events": [_ev("grade_documents", t, f"{len(rel)}/{len(docs)} chunks kept{note}", kept=ids)]}


def check_answerability(s):
    t = time.time()
    if not s["relevant"]:
        return {"answerable": False,
                "missing": "No text is indexed for this session. Upload a file again (indexing is lost if the server restarted).",
                "events": [_ev("check_answerability", t, "Nothing retrieved: index is empty")]}
    r = _json(f"Question: {s['question']}\n\nContext:\n{_ctx(s['relevant'])}\n\nCan the question be answered, "
              "at least in its main part, from this context? Return JSON {\"answerable\": true/false, "
              "\"missing\": \"what is absent, or empty\"}. Answer false only if the context clearly lacks the answer.")
    ok = str(r.get("answerable", True)).lower() not in ("false", "no", "0")
    return {"answerable": ok, "missing": r.get("missing", ""),
            "events": [_ev("check_answerability", t, "Answerable" if ok else "Context absent", missing=r.get("missing", ""))]}


def absence(s):
    t = time.time()
    msg = ("The uploaded documents do not contain enough information to answer this question. "
           f"Missing: {s.get('missing') or 'relevant content'}.")
    return {"answer": msg, "verdict": {"badge": "Context Absent", "level": "absent"},
            "events": [_ev("absence_flag", t, "Context absence flagged; generation skipped")]}


def generate(s):
    t = time.time()
    fb = f"\nA previous draft was rejected: {s['feedback']}. Fix this; use only the context." if s.get("feedback") else ""
    a = _llm().invoke(f"Answer using ONLY the context. Cite sources like [1]. If unsure, say so.{fb}\n\n"
                      f"Context:\n{_ctx(s['relevant'])}\n\nQuestion: {s['question']}").content
    n = s.get("retries", 0) + (1 if s.get("feedback") else 0)
    return {"answer": a, "retries": n, "events": [_ev("generate", t, f"Draft generated (attempt {n + 1})", draft=a)]}


def verify(s):
    """Dual-path verification, run in parallel."""
    t, ctx, a = time.time(), _ctx(s["relevant"]), s["answer"]

    def path_a():  # claim-level groundedness judge
        return _json(f"Context:\n{ctx}\n\nAnswer:\n{a}\n\nIs every claim in the answer supported by the context? "
                     "Return JSON {\"grounded\": bool, \"unsupported\": [claims]}.")

    def path_b():  # independent re-derivation, then semantic comparison
        ind = _llm().invoke(f"Answer concisely using ONLY this context.\n\n{ctx}\n\nQuestion: {s['question']}").content
        r = _json(f"Answer X:\n{a}\n\nAnswer Y:\n{ind}\n\nDo X and Y agree on the key facts? "
                  "Return JSON {\"agree\": bool, \"difference\": \"short note\"}.")
        return {**r, "independent": ind}

    with ThreadPoolExecutor(2) as ex:
        fa, fb = ex.submit(path_a), ex.submit(path_b)
        A, B = fa.result(), fb.result()
    grounded, agree = bool(A.get("grounded")), bool(B.get("agree"))
    if grounded and agree:
        v = {"badge": "High Agreement", "level": "high"}
    else:
        v = {"badge": "Path Divergence Detected", "level": "diverge"}
    v.update(grounded=grounded, agree=agree, unsupported=A.get("unsupported", []), difference=B.get("difference", ""))
    return {"verdict": v, "feedback": "; ".join(str(x) for x in (A.get("unsupported") or []) if x) or B.get("difference", ""),
            "events": [_ev("verify", t, f"Path A grounded={grounded} | Path B agree={agree} -> {v['badge']}")]}


def grade_answer(s):
    t = time.time()
    r = _json(f"Question: {s['question']}\nAnswer: {s['answer']}\n\nDoes the answer address the question? "
              "Return JSON {\"useful\": bool, \"score\": 1-5}.")
    v = {**s["verdict"], "useful": bool(r.get("useful")), "score": r.get("score")}
    return {"verdict": v, "events": [_ev("grade_answer", t, f"useful={v['useful']} score={v['score']}")]}


def route_answerable(s):
    return "generate" if s["answerable"] else "absence"


def route_after_grade(s):
    v = s["verdict"]
    bad = v["level"] != "high" or not v["useful"]
    return "generate" if bad and s.get("retries", 0) < MAX_RETRIES else END


def build():
    g = StateGraph(State)
    for n, f in [("retrieve", retrieve), ("grade_documents", grade_documents),
                 ("check_answerability", check_answerability), ("absence", absence),
                 ("generate", generate), ("verify", verify), ("grade_answer", grade_answer)]:
        g.add_node(n, f)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_edge("grade_documents", "check_answerability")
    g.add_conditional_edges("check_answerability", route_answerable, {"generate": "generate", "absence": "absence"})
    g.add_edge("absence", END)
    g.add_edge("generate", "verify")
    g.add_edge("verify", "grade_answer")
    g.add_conditional_edges("grade_answer", route_after_grade, {"generate": "generate", END: END})
    return g.compile()
