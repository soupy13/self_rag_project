"""Retrieval store. Default (SEMANTIC=0): in-memory BM25 only - tiny, instant, fits 512 MB.
SEMANTIC=1 adds ChromaDB + fastembed vectors built in the background (needs ~2 GB RAM)."""
import os, re, threading, uuid
from rank_bm25 import BM25Okapi

SEMANTIC = os.getenv("SEMANTIC", "0") == "1"
DATA_DIR = os.getenv("DATA_DIR", ".")
_mem, JOBS, _lock = {}, {}, threading.Lock()
_client = _model = None
_STOP = set("a an the is are was were of in on at to for and or what which who how many much does do did with by from this that it as be".split())
_tok = lambda s: [w for w in re.findall(r"\w+", s.lower()) if w not in _STOP] or re.findall(r"\w+", s.lower())


def _chroma():
    global _client
    if _client is None:
        import chromadb
        _client = chromadb.PersistentClient(path=os.path.join(DATA_DIR, "chroma_db"))
    return _client


def _col(sid):
    return _chroma().get_or_create_collection(f"s_{sid}", embedding_function=None)


def _embedder():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
                               threads=os.cpu_count() or 4, cache_dir=os.path.join(DATA_DIR, "models"))
    return _model


def _state(sid):
    if sid not in _mem:
        chunks = []
        if SEMANTIC:
            got = _col(sid).get(include=["documents", "metadatas"])
            chunks = [{"id": i, "text": d, **m} for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])]
        _mem[sid] = {"chunks": chunks, "bm25": None}
    return _mem[sid]


def _job(sid):
    return JOBS.setdefault(sid, {"done": 0, "total": 0, "pending": 0, "embed_pending": 0,
                                 "errors": [], "stage": "", "semantic": SEMANTIC})


def job_start(sid, name):
    j = _job(sid)
    j["pending"] += 1
    j["stage"] = f"Parsing {name}..."


def job_end(sid, name, err=None):
    j = _job(sid)
    j["pending"] -= 1
    j["stage"] = ""
    if err:
        j["errors"].append(f"{name}: {err}")


def add(sid, chunks):
    new = [{"id": uuid.uuid4().hex, "text": t, **m} for t, m in chunks]
    with _lock:
        st = _state(sid)
        st["chunks"] += new
        st["bm25"] = None
    j = _job(sid)
    j["total"] += len(new)
    if SEMANTIC:
        j["embed_pending"] += 1
        threading.Thread(target=_embed, args=(sid, new, j), daemon=True).start()


def _embed(sid, new, job, bs=256):
    try:
        model, col = _embedder(), _col(sid)
        for i in range(0, len(new), bs):
            b = new[i:i + bs]
            vecs = [v.tolist() for v in model.embed([c["text"] for c in b], batch_size=64)]
            col.add(ids=[c["id"] for c in b], documents=[c["text"] for c in b],
                    metadatas=[{"source": c["source"], "location": c["location"]} for c in b], embeddings=vecs)
            job["done"] += len(b)
    except Exception as e:
        job["errors"].append(f"semantic index: {e}")
    job["embed_pending"] -= 1


def query(sid, q, k):
    st = _state(sid)
    if not st["chunks"]:
        return []
    if st["bm25"] is None:
        st["bm25"] = BM25Okapi([_tok(c["text"]) for c in st["chunks"]])
    sc = st["bm25"].get_scores(_tok(q))
    order = sorted(range(len(sc)), key=lambda i: -sc[i])[:k * 2]
    ranked = [st["chunks"][i]["id"] for i in order]
    by_id = {c["id"]: c for c in st["chunks"]}
    vec = []
    if SEMANTIC:
        try:
            col = _col(sid)
            if col.count() > 0:
                qv = next(iter(_embedder().query_embed(q))).tolist()
                vec = col.query(query_embeddings=[qv], n_results=min(k * 2, col.count()))["ids"][0]
        except Exception as e:
            print("[store] vector search skipped:", e)
    score = {}
    for lst in (ranked, vec):  # reciprocal rank fusion
        for r, i in enumerate(lst):
            score[i] = score.get(i, 0) + 1 / (60 + r)
    top = sorted(score, key=score.get, reverse=True)[:k]
    return [{k2: by_id[i][k2] for k2 in ("text", "source", "location")} for i in top if i in by_id]


def sources(sid):
    counts = {}
    for c in _state(sid)["chunks"]:
        counts[c["source"]] = counts.get(c["source"], 0) + 1
    return counts


def job(sid):
    return JOBS.get(sid)


def reset(sid):
    if SEMANTIC:
        try:
            _chroma().delete_collection(f"s_{sid}")
        except Exception:
            pass
    _mem.pop(sid, None)
    JOBS.pop(sid, None)
