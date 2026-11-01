"""
smoke_api.py -- Stage 4C HTTP-layer checks. NOT RUN by its author. Run from EDI/validation (venv active):
    pip install httpx                       (TestClient needs it; one time)
    python smoke_api.py                     (500 particles; no uvicorn needed, the app is driven in-process)

Checks (each prints PASS/FAIL; exit code 1 if any FAIL):
  H1 discovery: GET /api/policies lists the default policies.
  H2 create: POST /api/episode returns the descriptor; a policy subset is honoured; unknown policy / bad obs_kind /
     extra field -> 422.
  H3 404 for an unknown episode on /step and /result.
  H4 invalid step bodies -> 422 (shock out of range, extra field, duration < 1, regime out of range) and none of them
     advances the episode.
  H5 /result before the end -> 409.
  H6 full episode over HTTP with a shock at step 10: every response passes validate_step_response, t = 0..N-1, done only at
     the end, shock event at t=10, a second shock -> 422 with t unchanged, stepping a finished episode -> 409.
  H7 /result: 6 policies, no contract errors, modified.shock_step == 10, and IS equals a direct LiveEpisode run (1e-9),
     which proves the JSON layer loses nothing.
  H8 contract violation -> 500 and the step is rolled back (retry succeeds at the same t).
  H9 runtime failure inside a policy -> 409 and the step is rolled back (retry succeeds at the same t).
  H10 session eviction: with MAX_SESSIONS patched to 2, the oldest episode is evicted (404).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
os.environ["IMM_PARTICLES"] = "500"          # must be set before api_server is imported (it reads the variable at import)

from fastapi.testclient import TestClient

import api_server
from api_server import app
from live_session import LiveEpisode, DEFAULT_POLICIES
import interfaces as I

SEED, PART = 1_000_000, 500
FAILS = []
client = TestClient(app)


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def new_episode(**body):
    r = client.post("/api/episode", json={"seed": SEED, **body})
    return r, (r.json().get("episode_id") if r.status_code == 200 else None)


def step(eid, body=None):
    return client.post(f"/api/episode/{eid}/step", json=body or {})


# ---- H1 / H2 / H3 ---------------------------------------------------------------------------
print("\n== discovery, creation, unknown ids")
r = client.get("/api/policies")
check("H1 /api/policies lists the defaults", r.status_code == 200 and r.json()["default"] == list(DEFAULT_POLICIES),
      str(r.status_code))

r, eid = new_episode()
d = r.json()
check("H2a create returns the descriptor", r.status_code == 200 and d["t"] == 0 and d["done"] is False
      and d["N"] > 0 and d["policies"] == list(DEFAULT_POLICIES), str({k: d.get(k) for k in ("N", "t", "done")}))
r, sub_id = new_episode(policies=["twap", "ce_oracle"])
check("H2b policy subset honoured", r.status_code == 200 and r.json()["policies"] == ["twap", "ce_oracle"])
sr = step(sub_id)
check("H2c subset episode steps and reports only those policies",
      sr.status_code == 200 and sorted(sr.json()["policies"]) == ["ce_oracle", "twap"])
check("H2d unknown policy -> 422", new_episode(policies=["nope"])[0].status_code == 422)
check("H2e bad obs_kind -> 422", new_episode(obs_kind="hawkes")[0].status_code == 422)
check("H2f extra field -> 422", new_episode(foo=1)[0].status_code == 422)
check("H3a unknown episode /step -> 404", step("ep-doesnotexist").status_code == 404)
check("H3b unknown episode /result -> 404", client.get("/api/episode/ep-doesnotexist/result").status_code == 404)

# ---- H4 / H5 --------------------------------------------------------------------------------
print("\n== invalid step bodies")
bad = {"shock out of range": {"inject_shock": 9},
       "extra field": {"foo": 1},
       "duration < 1": {"force_regime": {"regime": 1, "duration": 0}},
       "regime out of range": {"force_regime": {"regime": 7, "duration": 3}}}
for label, body in bad.items():
    check(f"H4 [{label}] -> 422", step(eid, body).status_code == 422)
r = step(eid)
check("H4e rejected requests did not advance the episode", r.status_code == 200 and r.json()["t"] == 0,
      f"(first accepted step has t={r.json().get('t')})")
check("H5 /result before the end -> 409", client.get(f"/api/episode/{eid}/result").status_code == 409)

# ---- H6 / H7 --------------------------------------------------------------------------------
print("\n== full episode over HTTP (shock at step 10)")
N = d["N"]
resp = [r.json()]                                   # step t=0 was taken above
second_shock_ok = None
while len(resp) < N:
    t = len(resp)
    r = step(eid, {"inject_shock": 3.0}) if t == 10 else step(eid)
    if r.status_code != 200:
        check(f"H6 step t={t} returned 200", False, f"(status {r.status_code}: {r.text[:150]})")
        break
    resp.append(r.json())
    if t == 10:
        r2 = step(eid, {"inject_shock": 2.0})
        second_shock_ok = (r2.status_code == 422)
errs = [e for x in resp for e in I.validate_step_response(x)]
check("H6a every response passes validate_step_response", not errs, str(errs[:2]))
check("H6b t runs 0..N-1 and done only at the end", [x["t"] for x in resp] == list(range(N))
      and [x["done"] for x in resp] == [False] * (N - 1) + [True])
check("H6c shock event reported at t=10", any(ev["type"] == "shock_injected" and ev["t"] == 10 for ev in resp[10]["events"]))
check("H6d second shock -> 422 and did not advance", second_shock_ok is True and len(resp) == N)
check("H6e stepping a finished episode -> 409", step(eid).status_code == 409)

r = client.get(f"/api/episode/{eid}/result")
res = r.json() if r.status_code == 200 else {}
check("H7a /result -> 200 with all policies", r.status_code == 200 and sorted(res.get("policies", {})) == sorted(DEFAULT_POLICIES))
check("H7b no contract errors, shock recorded",
      all(not v["contract_errors"] for v in res.get("policies", {}).values()) and res.get("modified", {}).get("shock_step") == 10)
direct = LiveEpisode(SEED, imm_particles=PART)
while not direct.done:
    direct.step(inject_shock=3.0) if direct.t == 10 else direct.step()
lg = direct.logs()
gap = max(abs(res["policies"][n]["IS"] - lg[n].IS) for n in lg) if res else float("inf")
check("H7c HTTP IS == direct LiveEpisode IS", gap < 1e-9, f"(max |dIS| {gap:.2e})")

# ---- H8 / H9 --------------------------------------------------------------------------------
print("\n== failure handling over HTTP")
r, eid2 = new_episode()
step(eid2); step(eid2)                              # two good steps first
ep = api_server._sessions[eid2]
t0 = ep.t

orig_validate = I.validate_step_response
I.validate_step_response = lambda resp: ["injected contract violation"]
try:
    r = step(eid2)
finally:
    I.validate_step_response = orig_validate
check("H8a contract violation -> 500", r.status_code == 500, f"(status {r.status_code})")
check("H8b state rolled back", ep.t == t0 and len(ep.obs_rows) == t0 and len(ep.beliefs) == t0)
r = step(eid2)
check("H8c retry succeeds at the same t", r.status_code == 200 and r.json()["t"] == t0)

t1 = ep.t
victim = ep.policies[-1]
victim_act = victim.act
def boom(x):
    raise RuntimeError("injected policy failure")
victim.act = boom
try:
    r = step(eid2)
finally:
    victim.act = victim_act
check("H9a policy runtime failure -> 409", r.status_code == 409, f"(status {r.status_code})")
check("H9b state rolled back", ep.t == t1 and len(ep.obs_rows) == t1 and all(len(h["v"]) == t1 for h in ep.hist.values()))
r = step(eid2)
check("H9c retry succeeds at the same t", r.status_code == 200 and r.json()["t"] == t1)

# ---- H10 ------------------------------------------------------------------------------------
print("\n== session eviction")
old_max = api_server.MAX_SESSIONS
api_server.MAX_SESSIONS = 2
try:
    ids = [new_episode()[1] for _ in range(3)]
finally:
    api_server.MAX_SESSIONS = old_max
check("H10a oldest session evicted", step(ids[0]).status_code == 404)
check("H10b newest sessions kept", step(ids[1]).status_code == 200 and step(ids[2]).status_code == 200)

print(f"\n{'ALL CHECKS PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)