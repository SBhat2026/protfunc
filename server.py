from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import torch.nn as nn
import pandas as pd
import esm
import joblib
import json
import os
import re
import warnings

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Download model files from HF Hub on first boot if not present
HF_REPO   = "SBhat2026/protfunc-models"
HF_FILES  = ["baseline_res.pth", "mlb_public_v1.pkl", "go_annotations_fixed.csv"]

def ensure_model_files():
    missing = [f for f in HF_FILES if not os.path.exists(os.path.join(BASE_DIR, f))]
    if not missing:
        return
    print(f"Downloading {len(missing)} file(s) from HuggingFace...")
    from huggingface_hub import hf_hub_download
    for fname in missing:
        print(f"  {fname}...")
        path = hf_hub_download(repo_id=HF_REPO, filename=fname,
                               local_dir=BASE_DIR, repo_type="model")
        print(f"  saved to {path}")

ensure_model_files()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "interface.html"))

# GO map
CSV_PATH = os.path.join(BASE_DIR, "go_annotations_fixed.csv")

def load_go_map():
    try:
        df = pd.read_csv(CSV_PATH)
        mapping = {}
        for _, row in df.iterrows():
            go_id    = str(row["GO Annotation"]).strip()
            raw_name = str(row.get("Gene Ontology (molecular function)", "Unknown"))
            mapping[go_id] = raw_name.split(" [")[0].strip()
        print(f"GO map: {len(mapping)} labels")
        return mapping
    except Exception as e:
        print(f"GO map error: {e}")
        return {}

go_map = load_go_map()
mlb    = joblib.load(os.path.join(BASE_DIR, "mlb_public_v1.pkl"))
NUM_LABELS = len(mlb.classes_)

thresholds_path = os.path.join(BASE_DIR, "artifacts", "per_label_thresholds.json")
thresholds = json.load(open(thresholds_path)) if os.path.exists(thresholds_path) else {}

# Model
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
model = RecoveredBaselineModel().to(device)
ckpt  = torch.load(os.path.join(BASE_DIR, "baseline_res.pth"), map_location=device)
sd    = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
model.load_state_dict(sd)
model.eval()

esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
esm_model = esm_model.to(device).eval()
batch_converter = alphabet.get_batch_converter()

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
            seq  = re.sub(r"\s+", "", blocks[i+1]) if i+1 < len(blocks) else ""
            if seq:
                names.append(name)
                seqs.append(seq)
            i += 2
        return list(zip(names, seqs))
    seqs = [l.strip() for l in text.splitlines() if l.strip()]
    return [(f"Sequence {i+1}", s) for i, s in enumerate(seqs)]

@app.post("/predict")
async def predict(request: ProteinRequest):
    entries = parse_sequences(request.sequence)
    results = []
    for name, sequence in entries:
        if len(sequence) > 2500:
            results.append({"name": name, "error": "Sequence too long (max 2500 aa)"})
            continue
        try:
            _, _, tokens = batch_converter([("p", sequence)])
            with torch.no_grad():
                rep  = esm_model(tokens.to(device), repr_layers=[6])["representations"][6]
                emb  = rep[0, 1:len(sequence)+1].mean(0)
                prob = torch.sigmoid(model(emb.unsqueeze(0))).squeeze()
            if prob.dim() == 0:
                prob = prob.unsqueeze(0)
            preds = []
            for i, p in enumerate(prob):
                pv = float(p)
                if pv >= float(thresholds.get(str(i), 0.5)):
                    go_id = mlb.classes_[i]
                    preds.append({"go_id": go_id, "name": go_map.get(go_id, go_id),
                                  "prob": round(pv, 3)})
            preds = sorted(preds, key=lambda x: x["prob"], reverse=True)[:12]
            results.append({"name": name, "sequence_length": len(sequence),
                            "predictions": preds, "n_above_threshold": len(preds)})
        except Exception as e:
            results.append({"name": name, "error": str(e)})
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)