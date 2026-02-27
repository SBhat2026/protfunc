---
title: Protfunc
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Protfunc (Insecta WebApp)
Protein function prediction using ESM-2 embeddings and a custom neural network.

## How it works
This site uses a FastAPI backend to process protein sequences through the ESM-2 model. 
It then maps those embeddings to GO (Gene Ontology) annotations using a pre-trained head.
