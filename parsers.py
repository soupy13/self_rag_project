"""Multi-format document parsing + chunking. Returns [(text, metadata)]."""
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def _pdf(p):
    from pypdf import PdfReader
    return [(pg.extract_text() or "", f"page {i}") for i, pg in enumerate(PdfReader(p).pages, 1)]


def _docx(p):
    import docx
    d = docx.Document(p)
    out = [("\n".join(x.text for x in d.paragraphs if x.text.strip()), "body")]
    for i, t in enumerate(d.tables, 1):
        rows = [" | ".join(c.text.strip() for c in r.cells) for r in t.rows]
        out.append(("\n".join(rows), f"table {i}"))
    return out


MAX_CHARS = 1800
MAX_ROWS = int(os.getenv("MAX_ROWS_PER_SHEET", 200000))


def _fmt(c):
    if c is None:
        return ""
    if isinstance(c, float) and c.is_integer():
        return str(int(c))
    return str(c).strip()


def _table_chunks(rows, label):
    """Stream rows -> large header-aware chunks, plus one SUMMARY chunk (row count, numeric stats)."""
    out, head, buf, size, start, n, blank, nrows = [], None, [], 0, 0, 0, 0, 0
    stats = {}
    def flush():
        nonlocal buf, size
        if buf:
            hdr = f"[{label}] columns: {', '.join(head)}\n"
            out.append((hdr + "\n".join(buf), f"{label} rows {start}-{n}"))
        buf, size = [], 0
    for r in rows:
        n += 1
        vals = [_fmt(c) for c in r]
        if not any(vals):
            blank += 1
            if blank >= 100:
                break
            continue
        blank = 0
        if head is None:
            head = [v or f"col{i + 1}" for i, v in enumerate(vals)]
            continue
        if n > MAX_ROWS:
            break
        nrows += 1
        for h, c in zip(head, r):
            if isinstance(c, (int, float)) and not isinstance(c, bool):
                st = stats.setdefault(h, [0, 0.0, c, c])
                st[0] += 1; st[1] += c; st[2] = min(st[2], c); st[3] = max(st[3], c)
        line = "; ".join(f"{h}={v}" for h, v in zip(head, vals) if v)
        if not buf:
            start = n
        buf.append(line)
        size += len(line) + 1
        if size >= MAX_CHARS:
            flush()
    flush()
    if head:
        num = "; ".join(f"{h}: count={s[0]}, sum={s[1]:.2f}, mean={s[1] / s[0]:.2f}, min={s[2]}, max={s[3]}"
                        for h, s in stats.items())
        txt = (f"[{label}] SUMMARY: {nrows} data rows. Columns: {', '.join(head)}. "
               f"Numeric column statistics: {num or 'none'}.")
        out.insert(0, (txt, f"{label} summary"))
    return out


def _xlsx(p):
    import openpyxl
    wb, out = openpyxl.load_workbook(p, read_only=True, data_only=True), []
    for ws in wb.worksheets:
        try:
            ws.reset_dimensions()
        except Exception:
            pass
        out += _table_chunks(ws.iter_rows(values_only=True), f"sheet '{ws.title}'")
    wb.close()
    return out


def _csv(p):
    import csv
    with open(p, newline="", encoding="utf-8", errors="ignore") as f:
        return _table_chunks(csv.reader(f), "csv")


def _pptx(p):
    from pptx import Presentation
    out = []
    for i, s in enumerate(Presentation(p).slides, 1):
        parts = []
        for sh in s.shapes:
            if sh.has_text_frame:
                parts.append(sh.text_frame.text)
            if getattr(sh, "has_table", False) and sh.has_table:
                parts += [" | ".join(c.text for c in r.cells) for r in sh.table.rows]
        if s.has_notes_slide:
            parts.append("Notes: " + s.notes_slide.notes_text_frame.text)
        out.append(("\n".join(x for x in parts if x.strip()), f"slide {i}"))
    return out


def _text(p):
    return [(open(p, encoding="utf-8", errors="ignore").read(), "text")]


HANDLERS = {".pdf": _pdf, ".docx": _docx, ".xlsx": _xlsx, ".xlsm": _xlsx, ".pptx": _pptx,
            ".txt": _text, ".md": _text, ".csv": _csv}
PRESIZED = {".xlsx", ".xlsm", ".csv"}


MAX_TOTAL = int(os.getenv("MAX_CHUNKS", 20000))


def parse_and_chunk(path, filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in HANDLERS:
        raise ValueError(f"Unsupported file type: {ext}")
    chunks = []
    for text, loc in HANDLERS[ext](path):
        if not text.strip():
            continue
        pieces = [text] if ext in PRESIZED else SPLITTER.split_text(text.strip())
        chunks += [(x, {"source": filename, "location": loc}) for x in pieces]
        if len(chunks) >= MAX_TOTAL:
            break
    return chunks[:MAX_TOTAL]
