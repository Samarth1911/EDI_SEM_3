import os
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_stage3c_multiseed.py")
text = open(path, encoding="utf-8").read()
open(path + ".bak", "w", encoding="utf-8").write(text)

def rep(old, new):
    global text
    assert text.count(old) == 1, f"anchor not found exactly once: {old!r}"
    text = text.replace(old, new)

rep('    ap.add_argument("--no-save-ckpt", action="store_true")\n',
    '    ap.add_argument("--no-save-ckpt", action="store_true")\n'
    '    ap.add_argument("--use-cache", action="store_true", help="replay cached beliefs for filtered arms")\n'
    '    ap.add_argument("--cache-root", default=None)\n')
rep('    from ddqn_train import run_episodes, paired, EVAL_BASE\n',
    '    from ddqn_train import run_episodes, paired, EVAL_BASE, TRAIN_BASE\n'
    '    from ddqn_train2 import VAL_BASE\n')
rep('    def build(spec, seed):\n', '    def build_real(spec, seed):\n')
rep('    test_seeds = range(EVAL_BASE, EVAL_BASE + a.eval_episodes)\n',
'''    def build(spec, seed):
        if not (a.use_cache and spec["arm"] == "filtered"):
            return build_real(spec, seed)
        from belief_cache import BeliefCache, make_cached_env
        single = spec.get("filt") == "single"
        cfg = EnvConfig(obs_mode="filtered", use_uncertainty=not spec.get("no_unc", False),
                        use_regime=not spec.get("no_regime", False), n_regimes=1 if single else 2,
                        reward_scale=a.reward_scale)
        root = a.cache_root or os.path.join(HERE, "outputs", "belief_cache")
        cache = BeliefCache(root, "single" if single else "imm", a.obs_kind, a.imm_particles,
                            [(TRAIN_BASE * (seed + 1), a.episodes), (VAL_BASE, a.n_val), (EVAL_BASE, a.eval_episodes)])
        return make_cached_env(cfg, cache)

    test_seeds = range(EVAL_BASE, EVAL_BASE + a.eval_episodes)
''')
open(path, "w", encoding="utf-8").write(text)
print("patched (backup: run_stage3c_multiseed.py.bak)")