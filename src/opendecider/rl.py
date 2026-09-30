"""Stage 2, "RLCD-like": outcome-only reward, used only where per-option labels do not exist.

This is OUR guess at RLCD (TypeSafe has not published it). Reward follows RLCR (Damani et al.
2025): correctness + Brier on the probability of the chosen option,
    r = R - (p_chosen - R)^2,   R in {0,1} = episode outcome.
Algorithm: GRPO (group-normalised advantages over G rollouts from the same start) or REINFORCE
with a running baseline. With full per-option labels, use train.py instead: the expected
Brier/log reward there equals the supervised loss, so RL adds nothing (§5.3).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .batching import Question, state_texts

MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


class GridEnv:
    """Multi-step environment: reach the goal in a small grid with walls; reward only at the end."""

    def __init__(self, size=5, walls=4, max_steps=10, seed=0):
        self.size, self.n_walls, self.max_steps = size, walls, max_steps
        self.rng = random.Random(seed)

    def reset(self):
        cells = [(r, c) for r in range(self.size) for c in range(self.size)]
        self.rng.shuffle(cells)
        self.pos, self.goal = cells[0], cells[1]
        self.walls = set(cells[2:2 + self.n_walls])
        self.t = 0
        return self.observe()

    def clone_start(self):
        e = GridEnv(self.size, self.n_walls, self.max_steps)
        e.pos, e.goal, e.walls, e.t = self.pos, self.goal, set(self.walls), 0
        return e

    def observe(self):
        rows = []
        for r in range(self.size):
            rows.append("".join("A" if (r, c) == self.pos else "G" if (r, c) == self.goal else
                                "#" if (r, c) in self.walls else "." for c in range(self.size)))
        state = {"grid": rows, "legend": "A=agent G=goal #=wall .=free", "steps_left": self.max_steps - self.t}
        opts = list(MOVES)
        self.rng.shuffle(opts)
        return state, Question("Which move gets the agent closer to the goal?", opts)

    def step(self, move: str):
        dr, dc = MOVES[move]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        if 0 <= r < self.size and 0 <= c < self.size and (r, c) not in self.walls:
            self.pos = (r, c)
        self.t += 1
        done = self.pos == self.goal or self.t >= self.max_steps
        return (None if done else self.observe()), float(self.pos == self.goal), done


@dataclass
class RLConfig:
    algo: str = "grpo"        # grpo | reinforce
    group: int = 4
    lr: float = 1e-5
    brier_weight: float = 1.0
    baseline_momentum: float = 0.9


def rlcr_reward(outcome: float, p_chosen: torch.Tensor, brier_weight: float = 1.0) -> torch.Tensor:
    return outcome - brier_weight * (p_chosen.detach() - outcome) ** 2


def rollout(model, env, max_steps):
    """Returns list of (log p(a_t), p(a_t)) and the final outcome."""
    obs = env.observe()
    traj, outcome = [], 0.0
    for _ in range(max_steps):
        state, q = obs
        mem = model.memory_from_ids(*model.backbone.pad(model.backbone.tokenize(state_texts([state]))))
        _, out = model.run([q], mem)
        logp = torch.log_softmax(out.logits[0, : len(q.options)], -1)
        a = torch.multinomial(logp.exp().detach(), 1).item()
        traj.append((logp[a], logp[a].exp()))
        obs, outcome, done = env.step(q.options[a])
        if done:
            break
    return traj, outcome


def rl_step(model, opt, env, cfg: RLConfig, state: dict | None = None) -> dict:
    state = state if state is not None else {"baseline": 0.0}
    env.reset()
    starts = [env.clone_start() for _ in range(cfg.group if cfg.algo == "grpo" else 1)]
    runs = [rollout(model, e, env.max_steps) for e in starts]
    # per-trajectory mean RLCR reward
    rewards = torch.stack([torch.stack([rlcr_reward(o, p, cfg.brier_weight) for _, p in tr]).mean()
                           for tr, o in runs])
    if cfg.algo == "grpo":
        adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
    else:
        adv = rewards - state["baseline"]
        state["baseline"] = cfg.baseline_momentum * state["baseline"] + (1 - cfg.baseline_momentum) * rewards.mean().item()
    loss = -torch.stack([a * torch.stack([lp for lp, _ in tr]).mean() for a, (tr, _) in zip(adv, runs)]).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
    opt.step()
    return {"loss": loss.item(), "reward": rewards.mean().item(),
            "success": sum(o for _, o in runs) / len(runs), **state}
