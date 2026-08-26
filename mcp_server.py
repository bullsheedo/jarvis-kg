"""JARVIS Knowledge Graph MCP server.

Exposes JARVIS's temporal knowledge graph to MCP clients (Hermes Agent, etc.)
Tools:
  - jarvis_remember(text)        -> ingest facts into the graph
  - jarvis_recall(query, k)      -> hybrid search over stored facts
  - jarvis_stats()               -> entity/edge/episode counts
  - jarvis_chat(message)         -> ask JARVIS (ox-alpha + graph memory)

Runs against the same FalkorDB Lite DB file the dashboard uses.
Auth: reads the dashboard token from data/dashboard.token (local file access).
"""
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

# load .env for OPENROUTER_API_KEY
from dotenv import load_dotenv
load_dotenv(BASE / ".env")

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("jarvis")

_graphiti = None

def _get_graphiti():
    global _graphiti
    if _graphiti is not None:
        return _graphiti
    from graphiti_core import Graphiti
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from redislite.async_falkordb_client import AsyncFalkorDB
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_client import OpenAIClient
    from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig
    from graphiti_core.cross_encoder.client import CrossEncoderClient
    from openai import AsyncOpenAI
    import asyncio

    OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]
    EMBED_DIM = 384

    class LocalEmbedder(EmbedderClient):
        class _Cfg(EmbedderConfig):
            embedding_dim: int = EMBED_DIM
        def __init__(self):
            self.config = self._Cfg(embedding_dim=EMBED_DIM)
            from fastembed import TextEmbedding
            self._model = TextEmbedding("BAAI/bge-small-en-v1.5")
        async def create(self, input_data):
            texts = [input_data] if isinstance(input_data, str) else list(input_data)
            vecs = [[float(x) for x in v] for v in self._model.embed(texts)]
            return vecs[0]
        async def create_batch(self, input_data_list):
            return [[float(x) for x in v] for v in self._model.embed(list(input_data_list))]

    class PassThroughCrossEncoder(CrossEncoderClient):
        async def rank(self, query, passages):
            return [(p, 0.0) for p in passages]

    oai = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)
    llm = OpenAIClient(config=LLMConfig(
        api_key=OPENROUTER_KEY, base_url="https://openrouter.ai/api/v1",
        model="google/gemini-2.5-flash", small_model="stealth/ox-alpha"), client=oai)

    db_path = BASE / "data" / "jarvis.db"
    client = AsyncFalkorDB(dbfilename=str(db_path))
    driver = FalkorDriver(falkor_db=client)
    _graphiti = Graphiti(graph_driver=driver, llm_client=llm,
                         embedder=LocalEmbedder(), cross_encoder=PassThroughCrossEncoder(),
                         store_raw_episode_content=False)
    return _graphiti

async def _remember(text: str, source: str):
    import datetime
    g = _get_graphiti()
    await g.build_indices_and_constraints()
    res = await g.add_episode(
        name=f"hermes-{datetime.datetime.now(datetime.UTC).strftime('%Y%m%d-%H%M%S')}",
        episode_body=text,
        source_description=source or "hermes-agent",
        reference_time=datetime.datetime.now(datetime.UTC),
    )
    facts = []
    for e in (res.edges or []):
        facts.append(e.fact)
    return {"ok": True, "facts": facts}

async def _recall(query: str, k: int):
    g = _get_graphiti()
    await g.build_indices_and_constraints()
    edges = await g.search(query, num_results=k)
    out = []
    for e in edges:
        out.append({
            "fact": e.fact,
            "valid_at": str(e.valid_at) if e.valid_at else None,
            "invalid_at": str(e.invalid_at) if e.invalid_at else None,
        })
    return out

@mcp.tool()
async def remember(text: str, source: str = "hermes-agent") -> dict:
    """Store facts into JARVIS's temporal knowledge graph. Pass conversation text; entities & relationships are extracted automatically."""
    return await _remember(text, source)

@mcp.tool()
async def recall(query: str, k: int = 8) -> dict:
    """Search JARVIS's memory. Returns temporal facts (with validity windows) relevant to the query."""
    results = await _recall(query, max(1, min(k, 25)))
    return {"results": results}

@mcp.tool()
async def stats() -> dict:
    """Get JARVIS knowledge graph statistics (entities, relations, episodes)."""
    g = _get_graphiti()
    async def one(cypher):
        records, header, _ = await g.graph_driver.execute_query(cypher) if hasattr(g, 'graph_driver') else (None,None,None)
        return records
    # Graphiti stores the driver on the instance; fall back across attribute names
    drv = getattr(g, "graph_driver", None) or getattr(g, "_graph_driver", None)
    if drv is None:
        # search works without driver access — derive counts from a broad recall instead
        edges = await g.search("all entities facts overview", num_results=25)
        return {"note": "driver not exposed; approximate", "sample_facts": [e.fact for e in edges[:10]]}
    async def count(cypher):
        records, header, _ = await drv.execute_query(cypher)
        return records[0]["c"] if records else 0
    ent = await count("MATCH (n:Entity) RETURN count(n) AS c")
    rel = await count("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS c")
    ep = await count("MATCH (n:Episodic) RETURN count(n) AS c")
    return {"entities": ent, "relations": rel, "episodes": ep}

if __name__ == "__main__":
    mcp.run(transport="stdio")
