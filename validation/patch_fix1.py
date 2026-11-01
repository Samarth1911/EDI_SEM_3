import os
H = os.path.dirname(os.path.abspath(__file__))

def patch(fname, edits):
    p = os.path.join(H, fname)
    t = open(p, encoding="utf-8").read()
    open(p + ".bak", "w", encoding="utf-8").write(t)
    for old, new in edits:
        assert t.count(old) == 1, f"{fname}: anchor not found exactly once: {old!r}"
        t = t.replace(old, new)
    open(p, "w", encoding="utf-8").write(t)
    print("patched", fname)

patch("export_agent_logs.py", [
    ('    ap.add_argument("--use-cache", action="store_true")\n', ''),
    ('    ranges = [(a.seed_base, a.n)]\n', ''),
    ('    logs["twap"] = run_logs(env0, seeds, TwapAgent(env0.twap_action), "twap")\n',
     '    ok = recs[0]["args"]["obs_kind"]\n'
     '    logs["twap"] = run_logs(env0, seeds, TwapAgent(env0.twap_action), "twap", obs_kind=ok)\n'),
    ('        env = build_env_for_rec(r, a.use_cache, ranges)\n', '        env = build_env_for_rec(r, False)   # real env: world_config needs env.sim.rp / env.filt.op\n'),
    ('        logs[name] = run_logs(env, seeds, agent, name)\n',
     '        logs[name] = run_logs(env, seeds, agent, name, obs_kind=r["args"]["obs_kind"])\n'),
])
patch("check_independence_hawkes.py", [
    ('            (real_res if e is real else cached_res).__setitem__(0, 0) if False else None\n', ''),
])