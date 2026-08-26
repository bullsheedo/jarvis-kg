import subprocess, json, time, sys

key = [l.strip().split('=',1)[1].strip('"\'') for l in open('/var/home/bazzite/.hermes/.env') if l.startswith('OPENROUTER_API_KEY')][0]

# Mimic Graphiti's actual extraction schema (simplified from graphiti_core prompts)
PROMPT = """Extract entities and relationships. You MUST respond with ONLY a JSON object with EXACTLY these keys: "extracted_entities" (list of {name, entity_type_id}) and "extracted_edge_types" (empty list ok).
Text: dev0id is building JARVIS on his Bazzite Linux machine. He prefers efficient local-first architectures."""

MODELS = [
    "google/gemini-2.5-flash",
    "z-ai/glm-5.2:free",
    "google/gemma-4-31b-it:free",
    "minimax/minimax-m3:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]

def test(model):
    t0 = time.time()
    r = subprocess.run(['curl','-s','-m','60','https://openrouter.ai/api/v1/chat/completions',
        '-H', f'Authorization: Bearer {key}', '-H','Content-Type: application/json',
        '-d', json.dumps({"model": model,
            "messages":[{"role":"user","content":PROMPT}],
            "response_format":{"type":"json_object"},
            "temperature":0})], capture_output=True, text=True, timeout=70)
    dt = time.time()-t0
    try:
        d = json.loads(r.stdout)
        if 'error' in d:
            return model, dt, False, str(d['error'])[:80]
        content = d['choices'][0]['message']['content']
        j = json.loads(content)
        ok = 'extracted_entities' in j
        ents = len(j.get('extracted_entities', [])) if isinstance(j.get('extracted_entities'), list) else -1
        return model, dt, ok, f"entities={ents} keys={list(j.keys())[:3]}"
    except Exception as ex:
        return model, dt, False, f"parse-fail: {str(ex)[:60]} raw={r.stdout[:80]}"

for m in MODELS:
    res = test(m)
    print(f"{'PASS' if res[2] else 'FAIL':4s} {res[1]:5.1f}s  {res[0]:45s} {res[3]}")
