"""JARVIS KG server — Graphiti temporal knowledge graph + embedded Kuzu + local fastembed embeddings."""
import os, asyncio, datetime, json
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from openai import AsyncOpenAI

from graphiti_core import Graphiti
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode

# ---- config ----
OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]
LLM_MODEL = "google/gemini-2.5-flash"       # extraction needs strict schema compliance
SMALL_MODEL = "stealth/ox-alpha"            # chat replies — free stealth reasoning model
# NOTE: ox-alpha fails Graphiti's structured extraction (~always missing 'extracted_entities'),
# so it's used only for /chat where freeform text is fine.
from graphiti_core.driver.falkordb_driver import FalkorDriver
from redislite.async_falkordb_client import AsyncFalkorDB

DB_PATH = str(Path(__file__).parent / "data" / "jarvis.db")
EMBED_DIM = 384                             # BAAI/bge-small-en-v1.5

# ---- local embeddings (fastembed, ONNX, no API cost) ----
from fastembed import TextEmbedding
_emb_model: TextEmbedding | None = None
def _embed(texts: list[str]) -> list[list[float]]:
    global _emb_model
    if _emb_model is None:
        _emb_model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return [[float(x) for x in v] for v in _emb_model.embed(texts)]

class LocalEmbedder(EmbedderClient):
    class _Cfg(EmbedderConfig):
        embedding_dim: int = EMBED_DIM
    def __init__(self):
        self.config = self._Cfg(embedding_dim=EMBED_DIM)
    async def create(self, input_data) -> list[float]:
        # Mirror graphiti's OpenAIEmbedder semantics: always return ONE flat vector
        texts = [input_data] if isinstance(input_data, str) else list(input_data)
        vecs = await asyncio.to_thread(_embed, texts)
        return vecs[0]
    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(_embed, list(input_data_list))

class PassThroughCrossEncoder(CrossEncoderClient):
    """Skip the extra LLM rerank call — hybrid search scores are good enough."""
    async def rank(self, query: str, passages: list) -> list:
        return [(p, 0.0) for p in passages]

# ---- OpenRouter LLM client (OpenAI-compatible) ----
oai = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)
from graphiti_core.llm_client.openai_client import OpenAIClient as OpenAIGenericClient
llm_client = OpenAIGenericClient(config=LLMConfig(
    api_key=OPENROUTER_KEY,
    base_url="https://openrouter.ai/api/v1",
    model=LLM_MODEL, small_model=SMALL_MODEL,
), client=oai)

falkordb_client = AsyncFalkorDB(dbfilename=DB_PATH)
kuzu_driver = FalkorDriver(falkor_db=falkordb_client)  # reused var name; it's the graph driver
graphiti = Graphiti(graph_driver=kuzu_driver, llm_client=llm_client,
                    embedder=LocalEmbedder(), cross_encoder=PassThroughCrossEncoder(),
                    store_raw_episode_content=False)

@asynccontextmanager
async def lifespan(app):
    await graphiti.build_indices_and_constraints()
    yield

app = FastAPI(title="JARVIS KG", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---- dashboard auth (Mark-LI style: 6-char session key -> bearer token) ----
import secrets as _secrets
DATA_DIR = Path(DB_PATH).parent
KEY_FILE = DATA_DIR / "dashboard.key"
TOKEN_FILE = DATA_DIR / "dashboard.token"

def _ensure_dashboard_creds():
    if not KEY_FILE.exists():
        key = _secrets.token_hex(3).upper()[:6]  # 6 hex chars
        token = _secrets.token_urlsafe(32)
        KEY_FILE.write_text(key)
        TOKEN_FILE.write_text(token)
        os.chmod(KEY_FILE, 0o600); os.chmod(TOKEN_FILE, 0o600)

def _check_auth(req: Request):
    token = (req.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
    if not token or token != TOKEN_FILE.read_text().strip():
        raise HTTPException(401, "unauthorized")

class LoginReq(BaseModel):
    key: str

@app.post("/login")
async def login(req: LoginReq):
    _ensure_dashboard_creds()
    if req.key.strip().upper() == KEY_FILE.read_text().strip():
        return {"token": TOKEN_FILE.read_text().strip()}
    raise HTTPException(401, "invalid key")

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return FileResponse(Path(__file__).parent / "static" / "login.html", media_type="text/html")

class IngestReq(BaseModel):
    text: str
    name: str | None = None
    source: str = "chat"

class SearchReq(BaseModel):
    query: str
    num_results: int = 10

name_map = {}

def edge_dto(e: EntityEdge) -> dict:
    return {"fact": e.fact, "source": name_map.get(e.source_node_uuid, e.source_node_uuid),
            "target": name_map.get(e.target_node_uuid, e.target_node_uuid),
            "relation": e.name, "uuid": e.uuid,
            "valid_at": str(e.valid_at) if e.valid_at else None,
            "invalid_at": str(e.invalid_at) if e.invalid_at else None}

def node_dto(n: EntityNode) -> dict:
    return {"name": n.name, "summary": getattr(n, "summary", ""), "uuid": n.uuid,
            "labels": getattr(n, "labels", [])}

@app.post("/ingest")
async def ingest(req: IngestReq, request: Request):
    _check_auth(request)
    t0 = datetime.datetime.now(datetime.UTC)
    try:
        res = await graphiti.add_episode(
            name=req.name or f"ep-{t0.strftime('%Y%m%d-%H%M%S')}",
            episode_body=req.text,
            source_description=req.source,
            reference_time=t0,
        )
        for n in (res.nodes or []):
            name_map[n.uuid] = n.name
        return {"ok": True, "facts_added": [edge_dto(e) for e in (res.edges or [])]}
    except Exception as ex:
        raise HTTPException(500, f"{type(ex).__name__}: {ex}")

@app.post("/search")
async def search(req: SearchReq, request: Request):
    _check_auth(request)
    try:
        edges = await graphiti.search(req.query, num_results=req.num_results)
        for uuid, name in await q("MATCH (n:Entity) RETURN n.uuid AS uuid, n.name AS name"):
            name_map[uuid] = name
        return {"results": [edge_dto(e) for e in edges]}
    except Exception as ex:
        raise HTTPException(500, f"{type(ex).__name__}: {ex}")

async def q(query: str):
    """Run a raw query; FalkorDriver returns (records, header, None)."""
    records, header, _ = await kuzu_driver.execute_query(query)
    return [tuple(r[h] for h in header) for r in records or []]

@app.get("/stats")
async def stats(request: Request):
    _check_auth(request)
    out = {}
    try:
        out["Entity"] = (await q("MATCH (n:Entity) RETURN count(n) AS c"))[0][0]
        out["RELATES_TO"] = (await q("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS c"))[0][0]
        out["Episodic"] = (await q("MATCH (n:Episodic) RETURN count(n) AS c"))[0][0]
    except Exception as ex:
        raise HTTPException(500, f"stats: {type(ex).__name__}: {ex}")
    return out

@app.get("/graph")
async def graph(request: Request, limit: int = 200):
    _check_auth(request)
    """Nodes+edges snapshot for the HUD visualization."""
    nodes, edges, seen = [], [], set()
    try:
        rows = await q(f"MATCH (n:Entity) RETURN n.uuid AS uuid, n.name AS name LIMIT {limit}")
        for uuid, name in rows:
            if uuid not in seen:
                seen.add(uuid); nodes.append({"id": uuid, "name": name})
        rows = await q(
            f"""MATCH (a:Entity)-[e:RELATES_TO]->(b:Entity)
                RETURN a.uuid AS s, b.uuid AS t, e.fact AS f LIMIT {limit*2}""")
        edge_seen = set()
        for s, t, fact in rows:
            key = (s, t, fact)
            if key not in edge_seen and s in seen and t in seen:
                edge_seen.add(key); edges.append({"source": s, "target": t, "label": fact})
    except Exception as ex:
        raise HTTPException(500, f"graph: {type(ex).__name__}: {ex}")
    return {"nodes": nodes, "edges": edges}

@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")

# ---- optional chat endpoint with graph-grounded context ----
class ChatReq(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatReq, request: Request):
    _check_auth(request)
    facts = (await search(SearchReq(query=req.message, num_results=6), request))["results"]
    memory = "\n".join(f"- {f['fact']}" for f in facts) or "(no relevant memories yet)"
    sys_prompt = ("You are JARVIS, a precise AI assistant. Use this temporal knowledge-graph "
                  "memory when relevant (respect validity — invalid_at means superseded):\n" + memory)
    async def gen():
        stream = await oai.chat.completions.create(
            model=SMALL_MODEL,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": req.message}],
            stream=True)
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta: yield delta
    return StreamingResponse(gen(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    Path(DB_PATH).parent.mkdir(exist_ok=True)
    uvicorn.run(app, host="127.0.0.1", port=8630)
