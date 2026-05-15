# SHL Conversational Assessment Recommender

A FastAPI service that takes a user from a vague hiring intent
("I'm hiring a Java developer") to a grounded shortlist of SHL
Individual Test Solutions through dialogue. It clarifies vague
queries, refines on constraint changes, compares named assessments
from catalog data, and refuses off-topic / injection attempts.

The agent is stateless: every `POST /chat` carries the full
conversation history and the service stores no per-conversation
state.

## Stack

* **FastAPI** for the HTTP surface — endpoint shapes match the spec
  exactly (`GET /health`, `POST /chat`).
* **Gemini 2.5 Flash-Lite** for the agent's planner + responder
  calls. The planner decides the conversational action; the
  responder grounds the recommendation in retrieved candidates.
* **Gemini Embedding 001** (3072-dim) for catalog vectors.
* **Hybrid retrieval**: cosine similarity on Gemini embeddings +
  a lightweight lexical signal (token / bigram overlap + exact name
  hits). The catalog is full of named products (e.g. `OPQ32r`,
  `ADEPT-15`) which a pure dense recall misses.
* **NumPy** for the index — 377 items × 3072 dims is trivial to keep
  in memory; FAISS would be overkill.

## Layout

```
app/
  catalog.py    # Assessment dataclass + JSON loader
  retriever.py  # Hybrid retriever over embeddings.npy
  llm.py        # Gemini wrappers (embed + chat-JSON)
  agent.py      # Planner -> Retriever -> Responder pipeline
  main.py       # FastAPI app
scripts/
  scrape_catalog.py  # Walks the SHL catalog and writes data/catalog.json
  build_index.py     # Reads catalog, writes embeddings.npy + meta JSON
tests/
  test_e2e.py        # Schema + behavior probes against a running service
data/
  catalog.json       # Scraped SHL Individual Test Solutions (377 items)
  embeddings.npy     # L2-normalized matrix (N x D)
  embeddings_meta.json
```

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Scrape the SHL catalog (once)
python scripts/scrape_catalog.py

# 2. Build the embedding index (free-tier-friendly, ~5 min)
export GEMINI_API_KEY=your_key
python scripts/build_index.py

# 3. Serve
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Deployed URL: https://shl-ai-agent-o2mn.onrender.com/docs
Quick sanity check:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}

curl -s -X POST http://127.0.0.1:8000/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hiring a Java developer who works with stakeholders"}]}'
```

## Agent design

For each turn the agent does at most two Gemini calls:

1. **Planner**. Reads the full message history + a static catalog
   primer (test-type letter legend, schema rules). Decides one of
   `clarify | recommend | refine | compare | refuse | smalltalk`,
   drafts a reply, and emits a retrieval query + a test-type filter
   string.
2. **Retriever**. For `recommend / refine / compare`, the hybrid
   retriever returns the top 20 candidates over the catalog.
3. **Responder**. Reads conversation + the candidate list and picks
   1-10 URLs by name. The service then validates every returned URL
   exists in the candidate set — URL hallucination is structurally
   impossible.

Defensive rails on top of the LLM:

* The very first user turn cannot produce a shortlist if the query
  is vague — handled server-side, independent of model output.
* If the responder picks zero URLs (or its JSON is broken), we fall
  back to the top retrieved candidates so the API never returns 0
  recs after committing to a recommendation.
* Recommendations are always capped at 1-10 items and forced through
  the public dataclass (`name`, `url`, `test_type`) — no extra keys
  ever leak.
* The history is sanitized before reaching the model: unknown roles
  / non-string content are dropped.

## Testing

`tests/test_e2e.py` is a small smoke suite that hits a running
server and covers the four behaviors, off-topic refusal, prompt
injection, and schema compliance. Run with:

```bash
python tests/test_e2e.py
```
