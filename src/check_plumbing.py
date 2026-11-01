import numpy as np
from ddqn import DDQNAgent, DDQNConfig
from ddqn_train2 import snapshot
from ddqn_train4 import _load_params
ag = DDQNAgent(7, 7, DDQNConfig(), 0)
snap = snapshot(ag)
_load_params(ag, [np.zeros_like(p) for p in ag.online.params()])
assert all(np.all(p == 0) for p in ag.online.params())
assert any(np.any(s != 0) for s in snap), "snapshot aliases live weights"
_load_params(ag, snap)
print("snapshot/EMA plumbing OK")
