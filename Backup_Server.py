from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from flask import Flask
import torch
import torch.nn as nn
import pandas as pd
import esm
import joblib
import json
import os
import re
import sys
import warnings
warnings.filterwarnings("ignore")

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def resource_path(rel_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, rel_path)
    return os.path.join(os.path.abspath("."), rel_path)

TEMPLATE_DIR = resource_path("templates")
STATIC_DIR = resource_path("static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)# =========================
# FASTAPI INIT
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory (css, html, etc.)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Serve interface.html at root
@app.get("/")
async def root():
    path = os.path.join(STATIC_DIR, "interface.html")
    print("Serving:", path)
    return FileResponse(path)

# =========================
# DEVICE
# =========================
device = torch.device("cpu")

# =========================
# LOAD GO MAP
# =========================
GO_JSON = "go_map.json"

if os.path.exists(GO_JSON):
    with open(GO_JSON) as f:
        go_map = json.load(f)
else:
    go_df = pd.read_csv(
        "/Users/siddhantbhat/Desktop/Research Files/go_annotations_fixed.csv",
        usecols=["GO Annotation", "Gene Ontology (molecular function)"]
    )

    go_df = go_df.dropna().drop_duplicates()

    def clean_go_name(text):
        return text.split("[")[0].strip()

    go_df["clean_name"] = go_df["Gene Ontology (molecular function)"].apply(clean_go_name)
    go_map = dict(zip(go_df["GO Annotation"], go_df["clean_name"]))

    with open(GO_JSON, "w") as f:
        json.dump(go_map, f)

# =========================
# LOAD LABEL BINARIZER
# =========================
mlb = joblib.load("mlb_public_v1.pkl")
NUM_LABELS = len(mlb.classes_)

# =========================
# MODEL ARCH
# =========================
class RecoveredBaselineModel(nn.Module):
    def __init__(self, input_dim=320, hidden_dim=1024, output_dim=NUM_LABELS, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        p = self.proj(x)
        h = h + p
        h = self.relu(self.fc2(h))
        h = self.drop(h)
        return self.out(h)

# =========================
# LOAD MODEL
# =========================
model = RecoveredBaselineModel().to(device)
model.load_state_dict(torch.load("baseline_state_dict.pth", map_location=device))
model.eval()

# =========================
# LOAD ESM
# =========================
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
esm_model = esm_model.to(device)
esm_model.eval()
batch_converter = alphabet.get_batch_converter()

# =========================
# INPUT SCHEMA
# =========================
class ProteinRequest(BaseModel):
    sequence: str

MAX_LEN = 2500
MAX_FUNCTIONS = 12   # <-- cap predictions per protein

# =========================
# FASTA PARSER
# =========================
def parse_sequences(text):
    text = text.strip()

    if text.startswith(">"):
        entries = re.split(r">.*\n", text)[1:]
        return [re.sub(r"\s+", "", e) for e in entries if e.strip()]

    return [line.strip() for line in text.splitlines() if line.strip()]

# =========================
# PREDICTION ROUTE
# =========================
@app.post("/predict")
async def predict(request: ProteinRequest):

    sequences = parse_sequences(request.sequence)
    all_results = []

    for sequence in sequences:

        if len(sequence) > MAX_LEN:
            all_results.append({
                "error": f"Sequence too long (>{MAX_LEN})"
            })
            continue

        _, _, tokens = batch_converter([("protein", sequence)])
        tokens = tokens.to(device)

        with torch.no_grad():
            out = esm_model(tokens, repr_layers=[6])
            emb = out["representations"][6][0, 1:len(sequence)+1].mean(0)

            logits = model(emb.unsqueeze(0))
            probs = torch.sigmoid(logits).squeeze()

        preds = []

        for i, p in enumerate(probs):
            prob = float(p)
            thr = thresholds.get(str(i), 0.5)
            if prob >= thr:
                go_id = mlb.classes_[i]
                preds.append({
                    "go_id": go_id,
                    "name": go_map.get(go_id, "Unknown"),
                    "prob": round(prob, 3)
                })

        # 🔥 sort by confidence
        preds = sorted(preds, key=lambda x: x["prob"], reverse=True)

        # 🔥 cap number of functions
        preds = preds[:MAX_FUNCTIONS]

        all_results.append({
            "sequence_length": len(sequence),
            "predictions": preds
        })

    return {"results": all_results}


# =========================
# RUN
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
