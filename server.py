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
mf_indices      = None
thresholds      = {}
NUM_LABELS      = 0


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, esm_model, batch_converter
    global mlb, go_map, mf_indices, thresholds, NUM_LABELS

    # Step 1: download missing files (network is ready here, unlike module load time)
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

    # Step 4: MF-only whitelist (mlb is now defined)
    mf_go_ids = {
        go_id for go_id, name in go_map.items()
        if name != go_id and not name.startswith("GO:")
    }
    if mf_go_ids:
        mf_indices = {i for i, go_id in enumerate(mlb.classes_) if go_id in mf_go_ids}
        print(f"MF-only filter: {len(mf_indices)} labels active")
    else:
        mf_indices = None
        print("MF filter not applied (go_names.json absent or empty)")

    # Step 5: per-label thresholds
    thresholds = load_thresholds()

    # Step 6: classifier
    class RecoveredBaselineModel(nn.Module):
        def __init__(self, input_dim=320, hidden_dim=1024, output_dim=NUM_LABELS, dropout=0.2):
            super().__init__()
            self.fc1  = nn.Linear(input_dim, hidden_dim)
            self.proj = nn.Linear(input_dim, hidden_dim)
            self.fc2  = nn.Linear(hidden_dim, hidden_dim)
            self.out  = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            h = self.relu(self.fc1(x))
            h = h + self.proj(x)
            h = self.relu(self.fc2(h))
            h = self.drop(h)
            return self.out(h)

    device = torch.device("cpu")
    _model = RecoveredBaselineModel().to(device)
    ckpt   = torch.load(os.path.join(BASE_DIR, "baseline_res.pth"), map_location=device)
    sd     = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    _model.load_state_dict(sd)
    _model.eval()
    model = _model
    print("Classifier loaded OK")

    # Step 7: ESM-2
    # THIS was the actual source of the curl error — esm.pretrained.esm2_t6_8M_UR50D()
    # internally runs curl/wget to download weights from huggingface.co at import time.
    # Being inside lifespan means it runs AFTER the container network stack is ready.
    import esm as esm_lib
    _esm_model, alphabet = esm_lib.pretrained.esm2_t6_8M_UR50D()
    esm_model       = _esm_model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    print("ESM-2 loaded OK")

    yield  # app is live

    print("Shutting down.")


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "interface.html"))


class ProteinRequest(BaseModel):
    sequence: str


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
    entries = parse_sequences(request.sequence)
    results = []
    device  = torch.device("cpu")

    for name, sequence in entries:
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

            preds  = []
            active = mf_indices if mf_indices else range(len(mlb.classes_))
            for i in active:
                pv = float(prob[i])
                if pv >= float(thresholds.get(str(i), 0.5)):
                    go_id = mlb.classes_[i]
                    preds.append({
                        "go_id": go_id,
                        "name":  go_map.get(go_id, go_id),
                        "prob":  round(pv, 3),
                    })
            preds = sorted(preds, key=lambda x: x["prob"], reverse=True)[:12]
            results.append({
                "name":              name,
                "sequence_length":   len(sequence),
                "predictions":       preds,
                "n_above_threshold": len(preds),
            })
        except Exception as e:
            results.append({"name": name, "error": str(e)})

    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)