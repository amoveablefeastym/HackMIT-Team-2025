#!/usr/bin/env python3
# extract_structured.py
# Turn input.txt into structured CSVs and a JSONL using LangExtract.
# Notes:
# - Uses Gemini via LangExtract by default (set LANGEXTRACT_API_KEY).
# - Adjust MODEL_ID via env LANGEXTRACT_MODEL_ID (default: gemini-2.5-flash).
# - Medical data here is for demo only; not for clinical use.

import os
import json
import csv
import textwrap
from pathlib import Path
from typing import Dict, Any, List, Iterable, Tuple, Optional

import langextract as lx
from dotenv import load_dotenv

# --------------------------
# Config & IO
# --------------------------
load_dotenv()
INPUT_PATH = Path("input.txt")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = os.getenv("LANGEXTRACT_MODEL_ID", "gemini-2.0-flash")
# keep it gentle for overloaded backends:
EXTRACTION_PASSES = int(os.getenv("EXTRACTION_PASSES", "1"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))
MAX_CHAR_BUFFER = int(os.getenv("MAX_CHAR_BUFFER", "800"))  # smaller chunks → fewer 503s


if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Missing {INPUT_PATH.resolve()}")

raw_text = INPUT_PATH.read_text(encoding="utf-8", errors="ignore")

# --------------------------
# Schema (wide but forgiving)
# Each extraction_class will be written to its own CSV with these columns.
# Always include: extraction_text, attributes_json, evidence_start, evidence_end
# --------------------------
SCHEMA: Dict[str, List[str]] = {
    "encounter": [
        "encounter_id", "patient_id", "datetime_start", "datetime_end",
        "provider", "location", "source_url"
    ],
    "patient": [
        "mrn", "name", "age", "sex", "dob", "race_ethnicity", "language"
    ],
    "problem": [
        "text_norm", "status", "onset_time", "resolved_time", "severity",
        "negated", "suspected", "code_system", "code"
    ],
    "symptom": [
        "text_norm", "status", "onset_time", "duration", "severity",
        "negated", "suspected", "body_site"
    ],
    "diagnosis": [
        "text_norm", "status", "suspected", "code_system", "code"
    ],
    "medication": [
        "drug_name", "rxnorm", "dose", "dose_unit", "route", "frequency",
        "prn", "indication", "start_time", "stop_time", "status", "negated"
    ],
    "allergy": [
        "substance", "rxnorm_or_snomed", "reaction", "severity", "status", "negated"
    ],
    "vital": [
        "type", "value", "unit", "position", "time"
    ],
    "lab": [
        "test_name", "loinc", "value", "unit", "ref_low", "ref_high",
        "interpretation", "time"
    ],
    "imaging_study": [
        "modality", "body_site", "finding", "impression", "code_system", "code", "time"
    ],
    "procedure": [
        "text_norm", "code_system", "code", "time", "laterality", "status"
    ],
    "vaccination": [
        "vaccine", "cvx_code", "date", "lot", "manufacturer"
    ],
    "social_history": [
        "category", "value", "qualifier", "start_time", "stop_time"
    ],
    "family_history": [
        "relation", "condition", "code_system", "code", "status"
    ],
    "risk_factor": [
        "factor", "status", "level"
    ],
    "assessment": [
        "statement", "problem_link"
    ],
    "plan_action": [
        "action", "target", "timeframe", "rationale", "problem_link"
    ],
    "referral": [
        "specialty", "reason", "status", "time"
    ],
    "follow_up": [
        "when", "reason", "modality"
    ],
    "order": [
        "order_type", "item", "priority", "status", "time"
    ],
    "device": [
        "device_name", "code_system", "code", "status"
    ],
    "smoking_status": [
        "status"
    ],
    "alcohol_use": [
        "status"
    ],
    "pain_score": [
        "score", "scale", "time"
    ],
    "functional_status": [
        "measure", "value", "unit", "time"
    ],
    "ros_finding": [
        "system", "finding", "negated"
    ],
    "physical_exam": [
        "system", "finding", "qualifier", "side", "severity", "time"
    ],
}

# universal columns appended to every table
UNIVERSAL = ["extraction_text", "attributes_json", "evidence_start", "evidence_end"]

# --------------------------
# Prompt & few-shot examples
# (Small but expressive; instruct exact-text capture + attributes.)
# --------------------------
PROMPT = textwrap.dedent("""\
    Extract structured clinical information directly from this outpatient transcript.
    Strictly use exact spans from the text for extraction_text fields (no paraphrasing).
    Do not fabricate information. If an attribute is missing, omit it.

    Extraction classes and attribute hints:
    - encounter: {encounter_id?, patient_id?, datetime_start?, datetime_end?, provider?, location?, source_url?}
    - patient: {mrn?, name?, age?, sex?, dob?, race_ethnicity?, language?}
    - problem: {text_norm?, status?, onset_time?, resolved_time?, severity?, negated?, suspected?, code_system?, code?}
    - symptom: {text_norm?, status?, onset_time?, duration?, severity?, negated?, suspected?, body_site?}
    - diagnosis: {text_norm?, status?, suspected?, code_system?, code?}
    - medication: {drug_name?, rxnorm?, dose?, dose_unit?, route?, frequency?, prn?, indication?, start_time?, stop_time?, status?, negated?}
    - allergy: {substance?, rxnorm_or_snomed?, reaction?, severity?, status?, negated?}
    - vital: {type?, value?, unit?, position?, time?}
    - lab: {test_name?, loinc?, value?, unit?, ref_low?, ref_high?, interpretation?, time?}
    - imaging_study: {modality?, body_site?, finding?, impression?, code_system?, code?, time?}
    - procedure: {text_norm?, code_system?, code?, time?, laterality?, status?}
    - vaccination: {vaccine?, cvx_code?, date?, lot?, manufacturer?}
    - social_history: {category?, value?, qualifier?, start_time?, stop_time?}
    - family_history: {relation?, condition?, code_system?, code?, status?}
    - risk_factor: {factor?, status?, level?}
    - assessment: {statement?, problem_link?}
    - plan_action: {action?, target?, timeframe?, rationale?, problem_link?}
    - referral: {specialty?, reason?, status?, time?}
    - follow_up: {when?, reason?, modality?}
    - order: {order_type?, item?, priority?, status?, time?}
    - device: {device_name?, code_system?, code?, status?}
    - smoking_status: {status?}
    - alcohol_use: {status?}
    - pain_score: {score?, scale?, time?}
    - functional_status: {measure?, value?, unit?, time?}
    - ros_finding: {system?, finding?, negated?}
    - physical_exam: {system?, finding?, qualifier?, side?, severity?, time?}

    Output rules:
    - Only extract facts that appear in the text.
    - Use 'negated': true if explicitly negated (e.g., "denies chest pain").
    - If uncertain speculation appears (e.g., "possible", "likely"), set 'suspected': true.
    - Prefer normalized attribute values when obvious (e.g., numeric vitals) but keep extraction_text exact.
""")

EXAMPLES = [
    lx.data.ExampleData(
        text="Vitals today: BP 128/82, HR 92 bpm, Temp 37.1 C. Meds: metformin 500 mg bid. Denies chest pain.",
        extractions=[
            lx.data.Extraction(
                extraction_class="vital",
                extraction_text="BP 128/82",
                attributes={"type": "blood_pressure", "value": "128/82", "unit": "mmHg", "time": "today"}
            ),
            lx.data.Extraction(
                extraction_class="vital",
                extraction_text="HR 92 bpm",
                attributes={"type": "heart_rate", "value": "92", "unit": "bpm", "time": "today"}
            ),
            lx.data.Extraction(
                extraction_class="vital",
                extraction_text="Temp 37.1 C",
                attributes={"type": "temperature", "value": "37.1", "unit": "C", "time": "today"}
            ),
            lx.data.Extraction(
                extraction_class="medication",
                extraction_text="metformin 500 mg bid",
                attributes={"drug_name": "metformin", "dose": "500", "dose_unit": "mg", "frequency": "bid"}
            ),
            lx.data.Extraction(
                extraction_class="symptom",
                extraction_text="Denies chest pain",
                attributes={"text_norm": "chest pain", "negated": True}
            ),
        ],
    ),
    lx.data.ExampleData(
        text="Assessment: Type 2 diabetes mellitus. Plan: increase metformin; order HbA1c.",
        extractions=[
            lx.data.Extraction(
                extraction_class="diagnosis",
                extraction_text="Type 2 diabetes mellitus",
                attributes={"code_system": "SNOMED", "code": "44054006"}  # example code
            ),
            lx.data.Extraction(
                extraction_class="plan_action",
                extraction_text="increase metformin",
                attributes={"action": "increase metformin", "problem_link": "diabetes"}
            ),
            lx.data.Extraction(
                extraction_class="order",
                extraction_text="order HbA1c",
                attributes={"order_type": "lab", "item": "HbA1c"}
            ),
        ],
    ),
]

# --------------------------
# Run LangExtract
# --------------------------
print(f"Running LangExtract with model: {MODEL_ID}")

from tenacity import retry, wait_exponential, wait_random, stop_after_attempt, retry_if_exception
import traceback

def _is_transient_error(e: Exception) -> bool:
    s = str(e).lower()
    # common transient signals from google.genai / proxies / gateways
    return any(tok in s for tok in ["503", "unavailable", "overloaded", "rate", "timeout", "temporar"])

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=30) + wait_random(0, 1),
    stop=stop_after_attempt(8),
    retry=retry_if_exception(_is_transient_error),
    reraise=True,
)
def _run_extract_once(model_id: str, text: str):
    return lx.extract(
        text_or_documents=text,
        prompt_description=PROMPT,
        examples=EXAMPLES,
        model_id=model_id,
        extraction_passes=EXTRACTION_PASSES,
        max_workers=MAX_WORKERS,
        max_char_buffer=MAX_CHAR_BUFFER,
    )

def extract_with_retry_and_fallback(text: str):
    models = [MODEL_ID]
    fb = os.getenv("FALLBACK_MODEL_ID")
    if fb and fb not in models:
        models.append(fb)

    last_err = None
    for m in models:
        print(f"LangExtract: trying model: {m}")
        try:
            return _run_extract_once(m, text)
        except Exception as e:
            last_err = e
            print(f"[warn] model {m} failed with: {e.__class__.__name__}: {e}")
            traceback.print_exc()
            print("---- trying next model (if any) ----")
    raise RuntimeError(f"All models failed. Last error: {last_err}")

result = extract_with_retry_and_fallback(raw_text)

# Save canonical JSONL (reviewable + visualizable)
jsonl_path = OUT_DIR / "extractions.jsonl"
lx.io.save_annotated_documents([result], output_name=str(jsonl_path.name), output_dir=str(OUT_DIR))
print(f"Saved JSONL: {jsonl_path.resolve()}")

# Optional: create a visualization HTML for manual QA
try:
    html_content = lx.visualize(str(jsonl_path))
    with open(OUT_DIR / "visualization.html", "w", encoding="utf-8") as f:
        f.write(getattr(html_content, "data", html_content))
    print(f"Saved HTML visualization: {(OUT_DIR / 'visualization.html').resolve()}")
except Exception as e:
    print(f"(non-fatal) Could not build visualization: {e}")

# --------------------------
# Normalize JSONL → tables
# We parse result lines, group by extraction_class, and write CSVs per class.
# --------------------------
def iter_extractions(jsonl_file: Path) -> Iterable[Tuple[str, str, Dict[str, Any], Optional[int], Optional[int]]]:
    """
    Yields: (extraction_class, extraction_text, attributes, evidence_start, evidence_end)
    Tries to be robust to slight schema differences.
    """
    with jsonl_file.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # Typical structure: obj.get("extractions") -> List[dict]
            ex_list = None
            for key in ("extractions", "entities", "results"):
                if isinstance(obj.get(key), list):
                    ex_list = obj[key]
                    break
            if not ex_list:
                # Some builds nest in obj["document"]["extractions"]
                doc = obj.get("document") or {}
                ex_list = doc.get("extractions", []) if isinstance(doc, dict) else []
            for ex in ex_list or []:
                cls = ex.get("extraction_class") or ex.get("class") or ex.get("label") or "unknown"
                text = ex.get("extraction_text") or ex.get("text") or ""
                attrs = ex.get("attributes") or {}
                # evidence offsets can appear in different fields
                start = None
                end = None
                spans = ex.get("spans") or ex.get("evidence") or []
                if isinstance(spans, list) and spans:
                    s0 = spans[0]
                    start = s0.get("start")
                    end = s0.get("end")
                yield cls, text, attrs, start, end

# Prepare writers for each known class + a catch-all
writers: Dict[str, Tuple[csv.DictWriter, Any]] = {}
files: Dict[str, Any] = {}

def make_writer(class_name: str, header_cols: List[str]) -> Tuple[csv.DictWriter, Any]:
    path = OUT_DIR / f"{class_name}.csv"
    f = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=header_cols, extrasaction="ignore")
    writer.writeheader()
    files[class_name] = f
    return writer, f

# Create writers for all declared classes
for cls, cols in SCHEMA.items():
    header = cols + UNIVERSAL
    writers[cls], _ = make_writer(cls, header)

# Also a catch-all for anything unexpected
writers["unknown"], _ = make_writer("unknown", ["class_name"] + UNIVERSAL)

# Write rows
for cls, text_span, attrs, start, end in iter_extractions(jsonl_path):
    row_base = {
        "extraction_text": text_span,
        "attributes_json": json.dumps(attrs, ensure_ascii=False),
        "evidence_start": start,
        "evidence_end": end,
    }
    if cls in SCHEMA:
        cols = {k: attrs.get(k, "") for k in SCHEMA[cls]}
        writers[cls].writerow({**cols, **row_base})
    else:
        writers["unknown"].writerow({"class_name": cls, **row_base})

# Close files
for f in files.values():
    f.close()

print("Wrote per-class CSVs to:", OUT_DIR.resolve())
print("Done.")
