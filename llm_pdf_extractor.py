
# LLM PDF Field Extraction — Stdlib Only (Fully Commented & Runnable)
# -------------------------------------------------------------------
# End-to-end pipeline to extract structured fields from PDFs without extra installs.
# Uses a best-effort stdlib PDF extractor + heuristics + a mock LLM (swap with a real one).
#
# How to run:
#   1) Place your PDF at /mnt/data/your_file.pdf (or update PDF_PATH below).
#   2) Run all cells. A JSON result will be saved next to the PDF.
#
# NOTE: This extractor is intentionally simple and may not handle all PDFs.
#       For robustness in production, use a mature PDF library and OCR for scans.

import os
import re
import zlib
import json
import difflib
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -------------------------
# 1) Schema configuration
# -------------------------

Schema = List[Dict[str, Any]]

FIELD_SCHEMA: Schema = [
    # Canonical field + aliases + regex/type hints
    {"name":"Permit Number","aliases":["Permit #","Permit Num","PermitNumber","Permit No","Permit ID","Permit No.","Permit Nbr"],"type":"string","regex":r"([A-Z0-9-]{3,})"},
    {"name":"Issue Date","aliases":["Issued On","Date of Issue","Issuance Date","Date"],"type":"date","regex":r"((?:\d{4}-\d{2}-\d{2})|(?:\d{2}/\d{2}/\d{4})|(?:\d{2}-\d{2}-\d{4}))"},
    {"name":"Applicant Name","aliases":["Owner","Applicant","Requestor","Name of Applicant"],"type":"string","regex":r"([A-Za-z ,.'-]{2,})"},
    {"name":"Site Address","aliases":["Project Address","Property Address","Location","Address"],"type":"string","regex":r"([0-9A-Za-z ,.#-]{5,})"}
]

def validate_value(field: Dict[str, Any], value: str) -> float:
    """Heuristic validator that returns a confidence score [0,1]."""
    score = 0.5
    value = (value or "").strip()
    if not value:
        return 0.0
    rgx = field.get("regex")
    if rgx and re.search(rgx, value):
        score += 0.4
    if field["type"] == "date" and re.search(r"\d{2,4}[-/]", value):
        score += 0.1
    return min(1.0, score)


# ---------------------------------------
# 2) PDF Text Extraction (Stdlib only)
# ---------------------------------------

def _extract_strings_from_TJ_array(arr: str) -> str:
    """Extract contiguous text from a PDF TJ array operand [ (a) 120 (b) ] TJ (simplified)."""
    out = []
    i, n = 0, len(arr)
    while i < n:
        if arr[i] == '(':
            i += 1
            buf = []
            depth = 1
            esc = False
            while i < n and depth > 0:
                c = arr[i]
                if esc:
                    buf.append(c)
                    esc = False
                else:
                    if c == '\\':
                        esc = True
                    elif c == '(':
                        depth += 1
                        buf.append(c)
                    elif c == ')':
                        depth -= 1
                        if depth == 0:
                            break
                        buf.append(c)
                    else:
                        buf.append(c)
                i += 1
            out.append(''.join(buf))
        else:
            i += 1
    return ''.join(out)

def _extract_strings_parens(s: str) -> str:
    """Extract text from tokens like '(Hello) Tj' (handles simple escapes/nesting)."""
    out = []
    i, n = 0, len(s)
    while i < n:
        if s[i] == '(':
            i += 1
            buf = []
            depth = 1
            esc = False
            while i < n and depth > 0:
                c = s[i]
                if esc:
                    buf.append(c)
                    esc = False
                else:
                    if c == '\\':
                        esc = True
                    elif c == '(':
                        depth += 1
                        buf.append(c)
                    elif c == ')':
                        depth -= 1
                        if depth == 0:
                            break
                        buf.append(c)
                    else:
                        buf.append(c)
                i += 1
            out.append(''.join(buf))
        else:
            i += 1
    return ' '.join(out)

def _try_inflate(data: bytes) -> bytes:
    """Attempt zlib decompression; return raw bytes if not FlateDecode."""
    try:
        return zlib.decompress(data)
    except Exception:
        return data

def extract_text_from_pdf_builtin(pdf_path: str) -> str:
    """
    Best-effort text extraction:
      - Read raw bytes
      - Find 'stream ... endstream' content blocks
      - Inflates with zlib (if possible)
      - Within inflated content, find BT ... ET blocks
      - Extract Tj/TJ strings
      - Normalize whitespace and return a single string
    """
    with open(pdf_path, 'rb') as f:
        raw = f.read()
    text_chunks = []
    # Find content streams
    for m in re.finditer(br'stream[\r\n]+(.*?)[\r\n]+endstream', raw, flags=re.DOTALL):
        inflated = _try_inflate(m.group(1))
        try:
            s = inflated.decode('latin-1', errors='ignore')
        except Exception:
            continue
        # Find text objects
        for bt in re.finditer(r'BT(.*?)ET', s, flags=re.DOTALL):
            body = bt.group(1)
            # ( ... ) Tj
            for tj in re.finditer(r'\((?:\\.|[^\)])*\)\s*Tj', body):
                text_chunks.append(_extract_strings_parens(tj.group(0)))
            # [ ... ] TJ
            for tja in re.finditer(r'\[(.*?)\]\s*TJ', body, flags=re.DOTALL):
                text_chunks.append(_extract_strings_from_TJ_array(tja.group(1)))
    # Fallback if nothing found
    if not text_chunks:
        try:
            s_all = raw.decode('latin-1', errors='ignore')
            for tj in re.finditer(r'\((?:\\.|[^\)])*\)\s*Tj', s_all):
                text_chunks.append(_extract_strings_parens(tj.group(0)))
        except Exception:
            pass
    joined = ' '.join(t.strip() for t in text_chunks if t and t.strip())
    joined = re.sub(r'\s+', ' ', joined).strip()
    return joined

def to_lines(text: str) -> List[str]:
    """Convert long text into pseudo-lines by simple splitting rules (for heuristics)."""
    text = re.sub(r'\s+', ' ', text).strip()
    return re.split(r'(?<=[\.:;])\s+|\s{2,}', text)


# ---------------------------------------
# 3) Heuristics (alias/regex/proximity)
# ---------------------------------------

def match_alias(canonical: str, aliases: List[str], text_header: str, cutoff: float = 0.7) -> bool:
    """True if text_header fuzzily matches canonical name or any alias."""
    candidates = [canonical] + aliases
    best = difflib.get_close_matches(text_header.lower(), [c.lower() for c in candidates], n=1, cutoff=cutoff)
    return len(best) > 0

def regex_extract(page_lines: List[str], field: Dict[str, Any]) -> List[Tuple[str, float, str]]:
    """Find 'Header: value' and next-line patterns; return regex-based candidates."""
    hits = []
    header_like = [field["name"]] + field["aliases"]
    header_pat = re.compile(r"|".join([re.escape(h) for h in header_like]), re.IGNORECASE)
    rgx = field.get("regex")
    for line in page_lines:
        if header_pat.search(line):
            after = line.split(":", 1)[-1] if ":" in line else line
            if rgx:
                m = re.search(rgx, after)
                if m:
                    hits.append((m.group(1).strip(), 0.6, "regex_header"))
    for i, line in enumerate(page_lines):
        if header_pat.search(line) and i + 1 < len(page_lines):
            candidate = page_lines[i + 1]
            if rgx:
                m = re.search(rgx, candidate)
                if m:
                    hits.append((m.group(1).strip(), 0.55, "regex_nearby"))
    return hits

def proximity_extract(page_lines: List[str], field: Dict[str, Any]) -> List[Tuple[str, float, str]]:
    """When formatting is messy, use alias match + next few tokens as a candidate value."""
    hits = []
    for line in page_lines:
        toks = [t.strip(": ").lower() for t in re.split(r"[\s:]+", line) if t.strip()]
        for idx, tok in enumerate(toks):
            if match_alias(field["name"], field["aliases"], tok, cutoff=0.8):
                tail = toks[idx + 1: idx + 4]
                if tail:
                    hits.append((" ".join(tail), 0.4, "proximity"))
                break
    return hits


# ---------------------------------------
# 4) LLM Interface (Mock)
# ---------------------------------------

@dataclass
class LLMResult:
    """Container for LLM responses; content is expected to be a JSON string."""
    content: str
    raw: Dict[str, Any] = field(default_factory=dict)

class LLMInterface:
    """Swap this out with your real LLM client (OpenAI, Azure, Bedrock, etc.)."""
    def extract(self, prompt: str) -> LLMResult:
        raise NotImplementedError

class MockLLM(LLMInterface):
    """Offline regex-based 'LLM' that returns JSON for the requested fields."""
    name = "MockLLM (offline)"
    def extract(self, prompt: str) -> LLMResult:
        m = re.search(r"SOURCE_TEXT\n---\n(.*)\n---\n", prompt, re.DOTALL)
        src = m.group(1) if m else ""
        data = {}
        permit = re.search(r"Permit(?:\s*Number|\s*#|\s*Num|Number|\s*No\.?|\s*ID|\s*Nbr)[^\w]*([A-Z0-9-]{3,})", src, re.IGNORECASE)
        if permit: data["Permit Number"] = permit.group(1)
        issue = re.search(r"(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4})", src)
        if issue: data["Issue Date"] = issue.group(1)
        appl = re.search(r"(?:Applicant|Owner)\s*[:\-]?\s*([A-Za-z ,.'-]{2,})", src)
        if appl: data["Applicant Name"] = appl.group(1).strip()
        addr = re.search(r"(?:Address|Location)\s*[:\-]?\s*([0-9A-Za-z ,.#-]{5,})", src)
        if addr: data["Site Address"] = addr.group(1).strip()
        return LLMResult(content=json.dumps(data))

def build_prompt(field_schema: Schema, source_text: str) -> str:
    """Construct a schema-aware prompt asking the LLM to return compact JSON only."""
    schema_desc = [f"- {f['name']} (type: {f['type']}, aliases: {', '.join(f['aliases'])})" for f in field_schema]
    schema_block = "\n".join(schema_desc)
    return f"""You are a careful information extraction model. Extract the requested fields.
Return only a compact JSON with keys exactly matching canonical field names.
If a value is missing, use an empty string.
SCHEMA
---
{schema_block}
---
SOURCE_TEXT
---
{source_text}
---
"""


# ---------------------------------------
# 5) Ensemble & Pipeline
# ---------------------------------------

def clean_tail(value: str) -> str:
    """Trim common spillover tokens from the end of a candidate value (e.g., 'Parcel ID')."""
    value = (value or "").strip()
    value = re.split(r"\b(Parcel ID|Zoning|Inspector|Contractor License|SCOPE)\b", value)[0].strip()
    value = re.sub(r"[\s,:;-]+$", "", value)
    return value

def merge_candidates(field: Dict[str, Any], candidates: List[Tuple[str, float, str]]) -> Dict[str, Any]:
    """Deduplicate, score, and choose the best candidate for a field."""
    if not candidates:
        return {"value": "", "score": 0.0, "method": ""}
    uniq, seen = [], set()
    for v, s, m in candidates:
        v = clean_tail(v)
        k = v.strip().lower()
        if k and k not in seen:
            uniq.append((v, s, m))
            seen.add(k)
    if not uniq:
        return {"value": "", "score": 0.0, "method": ""}
    uniq.sort(key=lambda x: x[1], reverse=True)
    best_v, best_s, best_m = uniq[0]
    best_s = max(best_s, validate_value(field, best_v))
    return {"value": best_v, "score": round(min(1.0, best_s), 3), "method": best_m}

def ensemble_extract_from_text(text: str, field_schema: Schema, llm: Optional[LLMInterface] = None) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """
    Run heuristics + LLM on plain text and merge results.
    Returns (result_dict, llm_name).
    """
    llm = llm or MockLLM()
    lines = to_lines(text)
    per_field_candidates = {f["name"]: [] for f in field_schema}
    for f in field_schema:
        per_field_candidates[f["name"]].extend(regex_extract(lines, f))
        per_field_candidates[f["name"]].extend(proximity_extract(lines, f))
    prompt = build_prompt(field_schema, text[:20000])
    try:
        res = llm.extract(prompt)
        llm_json = json.loads(res.content) if res and res.content.strip() else {}
    except Exception:
        llm_json = {}
    for f in field_schema:
        v = (llm_json.get(f["name"], "") if isinstance(llm_json, dict) else "")
        if v:
            per_field_candidates[f["name"]].append((v, 0.7, "llm"))
    merged = {f["name"]: merge_candidates(f, per_field_candidates[f["name"]]) for f in field_schema}
    return merged, getattr(llm, "name", llm.__class__.__name__)

def extract_fields_builtin(pdf_path: Optional[str] = None, raw_text: Optional[str] = None,
                           field_schema: Schema = None, llm: Optional[LLMInterface] = None) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """End-to-end entry: parse PDF (or use raw text) and run the ensemble pipeline."""
    assert (pdf_path or raw_text), "Provide either pdf_path or raw_text"
    field_schema = field_schema or FIELD_SCHEMA
    llm = llm or MockLLM()
    text = raw_text if raw_text else extract_text_from_pdf_builtin(pdf_path)
    return ensemble_extract_from_text(text, field_schema, llm)


# ---------------------------------------
# 6) Demo / Example usage
# ---------------------------------------

PDF_PATH = "/mnt/data/permit_extra_fields.pdf"  # change this to your file
if os.path.exists(PDF_PATH):
    preview = extract_text_from_pdf_builtin(PDF_PATH)[:300]
    result, llm_used = extract_fields_builtin(pdf_path=PDF_PATH)
    payload = {"llm_used": llm_used, "pdf": os.path.basename(PDF_PATH), "extracted_fields": result, "text_preview_first_300_chars": preview}
    with open("/mnt/data/extraction_result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print("\\nSaved JSON -> /mnt/data/extraction_result.json")
else:
    # Fallback: quick raw text demo
    raw = (
        "NORTHPORT BUILDING AUTHORITY\\n"
        "Permit # : ZZZ-2025-42\\n"
        "Owner : North Shore Homes\\n"
        "Date of Issue - 15-08-2025\\n"
        "Location: 1600 Lake View Rd, Northport, CA 96001\\n"
    )
    result, llm_used = ensemble_extract_from_text(raw, FIELD_SCHEMA)
    print("LLM used:", llm_used)
    print(json.dumps(result, indent=2))
