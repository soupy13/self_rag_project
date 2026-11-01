"""Flask server: upload, ask (SSE trace stream), reset."""
import json, os, shutil, tempfile, threading
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename
import graph, parsers, store

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
GRAPH = graph.build()


@app.get("/")
def index():
    return render_template("index.html")


def _ingest(sid, path, name, folder):
    err = None
    try:
        chunks = parsers.parse_and_chunk(path, name)
        if chunks:
            store.add(sid, chunks)
        else:
            err = "no text found (scanned PDF or empty file?)"
    except Exception as e:
        err = str(e)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        store.job_end(sid, name, err)


@app.post("/upload")
def upload():
    """Returns immediately; parsing/indexing runs in the background and the UI polls /sources."""
    sid, names = request.form["session"], []
    for f in request.files.getlist("files"):
        name = secure_filename(f.filename)
        folder = tempfile.mkdtemp()
        path = os.path.join(folder, name)
        f.save(path)
        store.job_start(sid, name)
        threading.Thread(target=_ingest, args=(sid, path, name, folder), daemon=True).start()
        names.append(name)
    return jsonify(queued=names)


@app.get("/sources")
def sources():
    sid = request.args["session"]
    return jsonify(sources=store.sources(sid), job=store.job(sid))


@app.get("/diagnose")
def diagnose():
    sid = request.args["session"]
    out = {"sources": store.sources(sid), "job": store.job(sid), "groq_key_set": bool(os.getenv("GROQ_API_KEY"))}
    try:
        out["groq_reply"] = graph._llm().invoke("Reply with the word OK").content
    except Exception as e:
        out["groq_error"] = str(e)
    return jsonify(out)


@app.post("/reset")
def reset():
    store.reset(request.json["session"])
    return jsonify(ok=True)


@app.post("/ask")
def ask():
    body = request.json

    def stream():
        final = {}
        try:
            for upd in GRAPH.stream({"question": body["question"], "session": body["session"]},
                                    stream_mode="updates"):
                for node, out in upd.items():
                    for e in out.get("events", []):
                        yield f"data: {json.dumps({'type': 'step', **e})}\n\n"
                    final.update({k: v for k, v in out.items() if k in ("answer", "verdict", "relevant")})
            srcs = [{"source": d["source"], "location": d["location"], "text": d["text"][:300]}
                    for d in final.get("relevant", [])]
            yield f"data: {json.dumps({'type': 'final', 'answer': final.get('answer'), 'verdict': final.get('verdict'), 'sources': srcs})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            if final.get("answer"):  # keep the draft visible, clearly marked unverified
                v = {"badge": "Unverified (error during verification)", "level": "diverge"}
                yield f"data: {json.dumps({'type': 'final', 'answer': final['answer'], 'verdict': v, 'sources': []})}\n\n"

    return Response(stream(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)
