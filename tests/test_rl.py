import torch

from opendecider.rl import GridEnv, RLConfig, rl_step, rlcr_reward
from tests.conftest import make


def test_rlcr_reward():
    assert rlcr_reward(1.0, torch.tensor(1.0)) == 1.0
    assert rlcr_reward(0.0, torch.tensor(0.0)) == 0.0
    assert rlcr_reward(1.0, torch.tensor(0.5)) == 0.75       # correct but unsure
    assert rlcr_reward(0.0, torch.tensor(0.9)) < -0.8        # wrong and confident


def test_grid_env_reaches_goal():
    e = GridEnv(size=3, walls=0, max_steps=10, seed=1)
    e.reset()
    e.pos, e.goal = (0, 0), (0, 1)
    _, r, done = e.step("right")
    assert r == 1.0 and done


def test_rl_step_updates_decision_module(bb):
    for algo in ("grpo", "reinforce"):
        m = make(bb, "v1").train()
        params = m.trainable_parameters()
        before = [p.detach().clone() for p in params]
        opt = torch.optim.AdamW(params, lr=1e-3)
        info = rl_step(m, opt, GridEnv(size=3, walls=1, max_steps=3, seed=0), RLConfig(algo=algo, group=3))
        assert "reward" in info
        assert all(p.grad is None for p in bb.parameters())
        if algo == "reinforce" or info["loss"] != 0:
            assert any(not torch.equal(a, b) for a, b in zip(before, params))
