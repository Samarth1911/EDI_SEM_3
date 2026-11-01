"""
apply_particle_patch.py -- Stage 4C amendment: opt-in particle cloud in the step response (for the Stage 5 per-regime cloud).
NOT RUN by its author. Run once from EDI/backend (venv active):   python apply_particle_patch.py

What it changes (all-or-nothing: if any anchor is not found exactly once, NOTHING is written; *.bak backups are made):
  live_session.py
    * class CloudIMM(IMMParticleFilter): copies the reweighted, PRE-resample posterior particles at the start of _predict().
      (IMMParticleFilter.update() calls _predict() right after building the belief, so after update() the filter's own arrays
      are already the prior for t+1. The copy is exactly the cloud the reported belief is computed from. No random numbers
      are drawn: filter results are identical to the plain class.)
    * LiveEpisode(..., particle_sample: int = 0): 0 = off (responses unchanged); m > 0 adds resp["particles"].
    * LiveEpisode._particle_block(t): uniform draw of m particles WITHOUT replacement per regime, from a stream of its own
      (seed, t) -- the filter's RNG is never touched, and a retried step returns the same sample.
  api_server.py
    * POST /api/episode accepts "particle_sample" (0..2000, default 0); > IMM_PARTICLES gives 422.

New optional top-level key of the step response (Stage 0 amendment; validate_step_response ignores unknown top-level keys):
  "particles": {"t", "n_per_regime", "alpha_ref", "kappa_ref",
                "regimes": [{"regime", "prob", "n_total", "alpha": [...], "kappa": [...], "w": [...]}, ...]}
  alpha, kappa in simulator units (divide by alpha_ref / kappa_ref for the filter's reference scale);
  prob = posterior regime probability; w = particle weight x n_total (1.0 = average weight within its regime), so a subsample
  of m particles is a UNIFORM sample of the weighted cloud: draw opacity/size from w, do not resample again.
"""
import os, re, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__))
LIVE, API = os.path.join(HERE, "live_session.py"), os.path.join(HERE, "api_server.py")

CLOUD = '''class CloudIMM(IMMParticleFilter):
    """IMMParticleFilter that keeps a copy of the reweighted, PRE-resample posterior of the latest update() when capture is on.
    update() calls _predict() (interaction -> resample -> propagate) right after building the belief, so the state at the
    start of _predict() is exactly the cloud the belief was computed from. Draws no random numbers."""
    capture = False
    last_posterior = None

    def _predict(self):
        if self.capture:
            self.last_posterior = (self.kappa.copy(), self.alpha.copy(), self.w.copy(), self.mu.copy())
        super()._predict()


'''

CTOR_CHECK = '''        self.particle_sample = int(particle_sample)
        if not 0 <= self.particle_sample <= imm_particles:
            raise ValueError(f"particle_sample must be in [0, {imm_particles}]")
        self.imm = CloudIMM(self.sp, self.op, self.rp, IMMParams(n_particles=imm_particles), seed=0)
        self.imm.capture = self.particle_sample > 0
'''

METHOD = '''    def _particle_block(self, t: int) -> dict:
        """Uniform sample (without replacement) of the filter's reweighted PRE-resample posterior particles at step t, per regime.
        Own RNG stream (seed, t): the filter's RNG is untouched and a retried step returns the same sample.
        w = particle weight * n_total (1.0 = average weight within its regime)."""
        kap, alp, w, mu = self.imm.last_posterior
        n, m = kap.shape[1], self.particle_sample
        rng = np.random.default_rng([self.seed, int(t), 0xC10D])
        regs = []
        for j in range(kap.shape[0]):
            idx = np.sort(rng.choice(n, size=m, replace=False))
            regs.append({"regime": j, "prob": float(mu[j]), "n_total": int(n), "alpha": alp[j, idx].tolist(),
                         "kappa": kap[j, idx].tolist(), "w": (w[j, idx] * n).tolist()})
        return {"t": int(t), "n_per_regime": int(m), "regimes": regs,
                "alpha_ref": getattr(self.op, "alpha_ref", None), "kappa_ref": getattr(self.op, "kappa_ref", None)}

'''


def read(p):
    with open(p, "r", newline="") as f:
        raw = f.read()
    return raw.replace("\r\n", "\n"), ("\r\n" if "\r\n" in raw else "\n")


def write(p, text, nl):
    with open(p, "w", newline="") as f:
        f.write(text.replace("\n", nl))


def sub1(text, pattern, fn, name, flags=0):
    new, n = re.subn(pattern, fn, text, flags=flags)
    if n != 1:
        raise SystemExit(f"anchor '{name}' found {n} times (need exactly 1): nothing was changed")
    return new


def patch_live(t):
    t = sub1(t, r'^PRESETS = \{"default"', lambda m: CLOUD + m.group(0), "PRESETS", re.M)
    t = sub1(t, r"guards: GuardConfig \| None = None\):", lambda m: "guards: GuardConfig | None = None, particle_sample: int = 0):", "ctor signature")
    t = sub1(t, r"^ *self\.imm = IMMParticleFilter\(self\.sp, self\.op, self\.rp, IMMParams\(n_particles=imm_particles\), seed=0\)\n",
             lambda m: CTOR_CHECK, "filter construction", re.M)
    t = sub1(t, r"\n([ \t]*)return I\.step_response\((.*?events=events\))",
             lambda m: (f"\n{m.group(1)}resp = I.step_response({m.group(2)}\n"
                        f"{m.group(1)}if self.particle_sample > 0:\n"
                        f"{m.group(1)}    resp[\"particles\"] = self._particle_block(t)\n"
                        f"{m.group(1)}return resp"), "step_response return", re.S)
    t = sub1(t, r"^    def logs\(self\) -> dict:", lambda m: METHOD + m.group(0), "logs()", re.M)
    return t


def patch_api(t):
    t = sub1(t, r"^    guards: Optional\[GuardRequest\] = None\n",
             lambda m: m.group(0) + "    particle_sample: int = Field(0, ge=0, le=2000)   # 0 = off; else particles per regime in each step response\n",
             "EpisodeRequest.guards", re.M)
    t = sub1(t, r"imm_particles=IMM_PARTICLES, guards=gcfg\)", lambda m: "imm_particles=IMM_PARTICLES, guards=gcfg, particle_sample=req.particle_sample)",
             "LiveEpisode call")
    return t


def main():
    lt, lnl = read(LIVE)
    at, anl = read(API)
    if "CloudIMM" in lt or "particle_sample" in at:
        sys.exit("already patched (CloudIMM / particle_sample found): nothing to do")
    lt2, at2 = patch_live(lt), patch_api(at)
    compile(lt2, LIVE, "exec"); compile(at2, API, "exec")          # syntax check before anything is written
    for p in (LIVE, API):
        if not os.path.exists(p + ".bak"):
            shutil.copy(p, p + ".bak")
    write(LIVE, lt2, lnl); write(API, at2, anl)
    print("patched live_session.py and api_server.py (backups: *.bak). Next: python ../validation/check_particles.py")


if __name__ == "__main__":
    main()