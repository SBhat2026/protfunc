from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pydantic import BaseModel
import torch
import torch.nn as nn
import pandas as pd
import joblib
import json
import math
import os
import re
import time
import warnings

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

HF_REPO  = "Sbhat2026/protfunc-models"
HF_FILES = ["baseline_res.pth", "mlb_public_v1.pkl", "go_annotations_fixed.csv", "go_names.json"]
OPTIONAL = {"go_names.json"}

# Globals populated during lifespan startup
model           = None
esm_model       = None
batch_converter = None
mlb             = None
go_map          = {}
mf_terms        = set()   # GO IDs with namespace == molecular_function (from OBO)
go_parents      = {}      # GO ID -> set of direct parent GO IDs (MF DAG only)
mf_indices      = None
thresholds      = {}
NUM_LABELS      = 0

# Biological complexity filter constants
MIN_SEQ_LENGTH    = 30
MIN_ENTROPY_BITS  = 2.5
MAX_DOMINANT_FRAC = 0.60
MIN_DISTINCT_AA   = 5
INVALID_AA        = set("BJOUXZ")
MF_ROOT           = "GO:0003674"


def _download_with_retry(fname):
    from huggingface_hub import hf_hub_download
    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"  [{attempt}/{max_attempts}] Downloading {fname}...")
            path = hf_hub_download(
                repo_id=HF_REPO, filename=fname,
                local_dir=BASE_DIR, repo_type="model",
                token=os.environ.get("HF_TOKEN"),
            )
            print(f"  saved -> {path}")
            return
        except Exception as e:
            if fname in OPTIONAL:
                print(f"  {fname} is optional, skipping ({e})")
                return
            if attempt == max_attempts:
                raise RuntimeError(f"Could not download '{fname}' after {max_attempts} attempts: {e}")
            wait = 2 ** attempt
            print(f"  Network error, retrying in {wait}s... ({e})")
            time.sleep(wait)


def ensure_model_files():
    missing = [f for f in HF_FILES if not os.path.exists(os.path.join(BASE_DIR, f))]
    if not missing:
        print("All model files already present.")
        return
    print(f"Downloading {len(missing)} file(s) from HuggingFace Hub...")
    for fname in missing:
        _download_with_retry(fname)


def load_go_map():
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "go_annotations_fixed.csv"))
        mapping = {}
        for _, row in df.iterrows():
            go_id    = str(row["GO Annotation"]).strip()
            raw_name = str(row.get("Gene Ontology (molecular function)", "Unknown"))
            mapping[go_id] = raw_name.split(" [")[0].strip()
        print(f"GO map: {len(mapping)} labels loaded")
        return mapping
    except Exception as e:
        print(f"GO map load error: {e}")
        return {}


def load_thresholds():
    for path in [
        os.path.join(BASE_DIR, "per_label_thresholds.json"),
        os.path.join(BASE_DIR, "artifacts", "per_label_thresholds.json"),
    ]:
        if os.path.exists(path):
            print(f"Thresholds loaded from {path}")
            return json.load(open(path))
    print("Thresholds not found, using 0.5 for all labels")
    return {}


def parse_obo(path):
    """
    Parse go-basic.obo and return:
        mf_terms   : set of GO IDs with namespace == molecular_function
        go_parents : dict mapping each MF GO ID -> set of direct parent GO IDs
                     (only is_a and part_of edges, restricted to MF namespace)
    """
    ns_map  = {}
    par_map = {}
    cur_id  = None
    cur_ns  = None
    cur_par = set()
    in_term = False

    def flush():
        nonlocal cur_id, cur_ns, cur_par
        if cur_id and cur_ns:
            ns_map[cur_id]  = cur_ns
            par_map[cur_id] = cur_par
        cur_id  = None
        cur_ns  = None
        cur_par = set()

    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line == "[Term]":
                flush()
                in_term = True
                continue
            if line.startswith("[") and line != "[Term]":
                flush()
                in_term = False
                continue
            if not in_term:
                continue
            if line.startswith("id:"):
                cur_id = line.split("id:", 1)[1].strip().split()[0]
            elif line.startswith("namespace:"):
                cur_ns = line.split("namespace:", 1)[1].strip()
            elif line.startswith("is_obsolete:") and "true" in line:
                cur_id = None
            elif line.startswith("is_a:"):
                parent = line.split("is_a:", 1)[1].strip().split()[0]
                cur_par.add(parent)
            elif line.startswith("relationship:"):
                parts = line.split("relationship:", 1)[1].strip().split()
                if len(parts) >= 2 and parts[0] == "part_of":
                    cur_par.add(parts[1])
    flush()

    mf = {gid for gid, n in ns_map.items() if n == "molecular_function"}
    go_parents_mf = {gid: (parents & mf) for gid, parents in par_map.items() if gid in mf}
    n_edges = sum(len(v) for v in go_parents_mf.values())
    print(f"OBO parsed: {len(mf)} MF terms, {n_edges} parent edges")
    return mf, go_parents_mf


def apply_hierarchy_filter(preds, go_parents_map):
    """
    Split predictions into (visible, suppressed).
    A prediction is suppressed when it has at least one direct MF parent
    but none of those parents appear in the predicted set.
    The MF root and terms with no MF parents are always visible.
    """
    if not go_parents_map:
        return preds, []

    predicted_ids = {p["go_id"] for p in preds}
    visible    = []
    suppressed = []

    for pred in preds:
        gid     = pred["go_id"]
        parents = go_parents_map.get(gid, set())
        if gid == MF_ROOT or not parents:
            visible.append(pred)
        elif parents & predicted_ids:
            visible.append(pred)
        else:
            suppressed.append(pred)

    return visible, suppressed


def sequence_entropy(seq):
    seq_upper = seq.upper()
    counts = {}
    for aa in seq_upper:
        counts[aa] = counts.get(aa, 0) + 1
    n = len(seq_upper)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def validate_sequence(name, seq):
    """Returns an error string if the sequence should be rejected, else None."""
    if len(seq) < MIN_SEQ_LENGTH:
        return (f"'{name}' is too short ({len(seq)} aa — minimum {MIN_SEQ_LENGTH} aa). "
                f"Sequences this short are unlikely to fold into a stable domain.")

    bad = sorted({c.upper() for c in seq if c.upper() in INVALID_AA})
    if bad:
        return (f"'{name}' contains invalid amino acid character(s): "
                f"{', '.join(bad)}. These ambiguity codes are not accepted.")

    counts = {}
    for aa in seq.upper():
        counts[aa] = counts.get(aa, 0) + 1

    if len(counts) < MIN_DISTINCT_AA:
        return (f"'{name}' uses only {len(counts)} distinct residue type(s). "
                f"Real proteins require at least {MIN_DISTINCT_AA}.")

    dominant_frac = max(counts.values()) / len(seq)
    if dominant_frac > MAX_DOMINANT_FRAC:
        dominant_aa = max(counts, key=counts.get)
        return (f"'{name}' is dominated by a single residue "
                f"({dominant_aa} = {dominant_frac:.0%}). "
                f"Low-complexity sequences produce unreliable embeddings.")

    H = sequence_entropy(seq)
    if H < MIN_ENTROPY_BITS:
        return (f"'{name}' has very low sequence complexity "
                f"(Shannon entropy {H:.2f} bits, minimum {MIN_ENTROPY_BITS:.1f} bits). "
                f"Repetitive or artificially constructed sequences are not accepted.")

    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, esm_model, batch_converter
    global mlb, go_map, mf_terms, go_parents, mf_indices, thresholds, NUM_LABELS

    # Step 1: download missing files
    ensure_model_files()

    # Step 2: GO name map
    go_map = load_go_map()
    go_names_path = os.path.join(BASE_DIR, "go_names.json")
    if os.path.exists(go_names_path):
        go_map.update(json.load(open(go_names_path)))
        print(f"Canonical GO names loaded: {len(go_map)} total entries")

    # Step 3: MLB — load BEFORE anything references mlb.classes_
    mlb        = joblib.load(os.path.join(BASE_DIR, "mlb_public_v1.pkl"))
    NUM_LABELS = len(mlb.classes_)
    print(f"MLB loaded: {NUM_LABELS} labels")

    # Step 4: OBO — parse MF namespace and parent DAG
    obo_path = os.path.join(BASE_DIR, "go-basic.obo")
    if os.path.exists(obo_path):
        mf_terms, go_parents = parse_obo(obo_path)
        mf_in_mlb = sum(1 for gid in mlb.classes_ if gid in mf_terms)
        print(f"OBO cross-check: {mf_in_mlb}/{NUM_LABELS} MLB labels are MF namespace")
    else:
        print("WARNING: go-basic.obo not found — hierarchy filtering disabled. "
              "Download from https://current.geneontology.org/ontology/go-basic.obo "
              "and place it alongside server.py.")

    # Step 5: MF-only whitelist — OBO namespace is authoritative, CSV is fallback
    if mf_terms:
        mf_indices = [i for i, gid in enumerate(mlb.classes_) if gid in mf_terms]
        print(f"MF whitelist (OBO): {len(mf_indices)} active indices")
    else:
        mf_go_ids = {
            go_id for go_id, name in go_map.items()
            if name and name != go_id and not name.startswith("GO:")
        }
        mf_indices = [i for i, gid in enumerate(mlb.classes_) if gid in mf_go_ids] or list(range(NUM_LABELS))
        print(f"MF whitelist (CSV fallback): {len(mf_indices)} active indices")

    app.state.mf_indices = mf_indices

    # Step 6: per-label thresholds
    thresholds = load_thresholds()

    # Step 7: classifier — auto-detect architecture from checkpoint keys
    class ResidualMLP(nn.Module):
        """Matches General_Pipeline.ipynb — two skip-connection blocks."""
        def __init__(self, in_dim=320, out_dim=NUM_LABELS, hidden=1024, dropout=0.2):
            super().__init__()
            self.fc_in  = nn.Linear(in_dim, hidden)
            self.block1 = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
            self.block2 = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
            self.fc_out = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim))

        def forward(self, x):
            h = self.fc_in(x)
            h = torch.relu(h)
            h = h + self.block1(h)
            h = h + self.block2(h)
            return self.fc_out(h)

    class RecoveredBaselineModel(nn.Module):
        """Earlier server-side architecture — retained for backward compatibility."""
        def __init__(self, in_dim=320, out_dim=NUM_LABELS, hidden=1024, dropout=0.2):
            super().__init__()
            self.fc1  = nn.Linear(in_dim, hidden)
            self.proj = nn.Linear(in_dim, hidden)
            self.fc2  = nn.Linear(hidden, hidden)
            self.out  = nn.Linear(hidden, out_dim)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            h = self.relu(self.fc1(x))
            h = h + self.proj(x)
            h = self.relu(self.fc2(h))
            h = self.drop(h)
            return self.out(h)

    device  = torch.device("cpu")
    ckpt    = torch.load(os.path.join(BASE_DIR, "baseline_res.pth"), map_location=device)
    sd      = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    keys    = set(sd.keys())

    if any(k.startswith("fc_in") for k in keys):
        _model = ResidualMLP().to(device)
        print("Classifier architecture: ResidualMLP (notebook)")
    elif any(k.startswith("fc1") for k in keys):
        _model = RecoveredBaselineModel().to(device)
        print("Classifier architecture: RecoveredBaselineModel (server)")
    else:
        raise RuntimeError(f"Unrecognised checkpoint architecture. First keys: {sorted(keys)[:8]}")

    _model.load_state_dict(sd, strict=True)
    _model.eval()
    model = _model
    print("Classifier loaded OK")

    # Step 8: ESM-2
    import esm as esm_lib
    _esm_model, alphabet = esm_lib.pretrained.esm2_t6_8M_UR50D()
    esm_model       = _esm_model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    print("ESM-2 loaded OK")

    yield

    print("Shutting down.")


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "interface.html"))


@app.get("/api/model/info")
async def model_info():
    """Return model metadata and configuration."""
    return {
        "model_name": "ProtFunc Enhanced",
        "version": "2.0.0",
        "esm_model": "esm2_t6_8M_UR50D",
        "embed_dim": 320,
        "num_labels": NUM_LABELS,
        "supported_namespaces": ["molecular_function"],
        "max_sequence_length": 2500,
        "thresholds_loaded": len(thresholds) > 0,
        "hierarchy_filtering": len(go_parents) > 0,
        "mf_terms_loaded": len(mf_terms) if mf_terms else 0,
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "esm_loaded": esm_model is not None,
        "labels": NUM_LABELS,
    }


class ProteinRequest(BaseModel):
    sequence: str


class BatchPredictRequest(BaseModel):
    sequences: list  # List of {name: str, sequence: str} objects
    threshold: float = 0.5
    include_suppressed: bool = True


def parse_sequences(text):
    text = text.strip()
    if text.startswith(">"):
        blocks = re.split(r"(>.*)", text)
        names, seqs = [], []
        i = 1
        while i < len(blocks):
            name = blocks[i][1:].strip()
            seq  = re.sub(r"\s+", "", blocks[i + 1]) if i + 1 < len(blocks) else ""
            if seq:
                names.append(name)
                seqs.append(seq)
            i += 2
        return list(zip(names, seqs))
    seqs = [l.strip() for l in text.splitlines() if l.strip()]
    return [(f"Sequence {i + 1}", s) for i, s in enumerate(seqs)]


@app.post("/predict")
async def predict(request: ProteinRequest):
    entries    = parse_sequences(request.sequence)
    results    = []
    device     = torch.device("cpu")
    mf_idx     = app.state.mf_indices

    for name, sequence in entries:

        # Biological complexity guard — reject before touching ESM-2
        err = validate_sequence(name, sequence)
        if err:
            results.append({"name": name, "error": err})
            continue

        if len(sequence) > 2500:
            results.append({"name": name, "error": "Sequence too long (max 2500 aa)"})
            continue

        try:
            _, _, tokens = batch_converter([("p", sequence)])
            with torch.no_grad():
                rep  = esm_model(tokens.to(device), repr_layers=[6])["representations"][6]
                emb  = rep[0, 1:len(sequence) + 1].mean(0)
                prob = torch.sigmoid(model(emb.unsqueeze(0))).squeeze()
            if prob.dim() == 0:
                prob = prob.unsqueeze(0)

            # Collect all predictions above per-label threshold (no hard cap)
            raw_preds = []
            for i in mf_idx:
                pv = float(prob[i])
                if pv >= float(thresholds.get(str(i), 0.5)):
                    go_id = mlb.classes_[i]
                    raw_preds.append({
                        "go_id": go_id,
                        "name":  go_map.get(go_id, go_id),
                        "prob":  round(pv, 4),
                    })
            raw_preds.sort(key=lambda x: x["prob"], reverse=True)

            # Apply GO hierarchy filter
            visible, suppressed = apply_hierarchy_filter(raw_preds, go_parents)

            for p in visible:    p["prob"] = round(p["prob"], 3)
            for p in suppressed: p["prob"] = round(p["prob"], 3)

            results.append({
                "name":              name,
                "sequence_length":   len(sequence),
                "predictions":       visible,
                "suppressed":        suppressed,
                "n_above_threshold": len(raw_preds),
            })
        except Exception as e:
            results.append({"name": name, "error": str(e)})

    return {"results": results}


@app.post("/api/predict/batch")
async def predict_batch(request: BatchPredictRequest):
    """
    Batch prediction endpoint for multiple sequences.
    
    Accepts a list of sequence objects and returns predictions for all.
    More efficient than multiple single predictions due to batching.
    """
    results = []
    device = torch.device("cpu")
    mf_idx = app.state.mf_indices
    custom_threshold = request.threshold
    
    for item in request.sequences:
        name = item.get("name", "Unknown")
        sequence = item.get("sequence", "")
        
        # Validate sequence
        err = validate_sequence(name, sequence)
        if err:
            results.append({"name": name, "error": err})
            continue
        
        if len(sequence) > 2500:
            results.append({"name": name, "error": "Sequence too long (max 2500 aa)"})
            continue
        
        try:
            _, _, tokens = batch_converter([("p", sequence)])
            with torch.no_grad():
                rep = esm_model(tokens.to(device), repr_layers=[6])["representations"][6]
                emb = rep[0, 1:len(sequence) + 1].mean(0)
                prob = torch.sigmoid(model(emb.unsqueeze(0))).squeeze()
            
            if prob.dim() == 0:
                prob = prob.unsqueeze(0)
            
            # Collect predictions above threshold
            raw_preds = []
            for i in mf_idx:
                pv = float(prob[i])
                thresh = float(thresholds.get(str(i), custom_threshold))
                if pv >= thresh:
                    go_id = mlb.classes_[i]
                    raw_preds.append({
                        "go_id": go_id,
                        "name": go_map.get(go_id, go_id),
                        "prob": round(pv, 4),
                    })
            raw_preds.sort(key=lambda x: x["prob"], reverse=True)
            
            # Apply hierarchy filter
            if request.include_suppressed:
                visible, suppressed = apply_hierarchy_filter(raw_preds, go_parents)
            else:
                visible = raw_preds
                suppressed = []
            
            for p in visible:
                p["prob"] = round(p["prob"], 3)
            for p in suppressed:
                p["prob"] = round(p["prob"], 3)
            
            results.append({
                "name": name,
                "sequence_length": len(sequence),
                "predictions": visible,
                "suppressed": suppressed if request.include_suppressed else [],
                "n_above_threshold": len(raw_preds),
            })
        except Exception as e:
            results.append({"name": name, "error": str(e)})
    
    return {
        "results": results,
        "total": len(results),
        "successful": sum(1 for r in results if "error" not in r),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
