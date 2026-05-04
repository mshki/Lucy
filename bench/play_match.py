"""Head-to-head match between two HU-NLHE-FCPA bots.

The match is played as N *duplicate pairs*: each unique deck (controlled by a
seeded RNG) is played twice, with the bots in swapped seats the second time.
This eliminates card-luck variance for free (~30–50% reduction). The reported
metric is **mbb/hand** (milli-big-blinds per hand) for ``bot_a``, with a 95%
confidence interval over per-pair averages.

Bots are constructed via factory callables so that:
  - Lucy bots can spin up their own subprocess per (player_id, model) pair.
  - OpenSpiel bots can share a single in-memory solver but bind a fresh
    sampling RNG per side.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

import numpy as np
import pyspiel  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.common import (BIG_BLIND, HU_NLHE_FCPA_GAMEDEF,  # noqa: E402
                          NUM_PLAYERS)
from bench.lucy_bot import LucyBot  # noqa: E402
from bench.openspiel_bot import (AlwaysFoldBot, OpenSpielMCCFRBot,  # noqa: E402
                                 UniformRandomBot, load_solver)


# ---------------------------------------------------------------------------
# Bot factories. Each factory takes (player_id, rng) and returns a pyspiel.Bot.
# ---------------------------------------------------------------------------

def make_random_factory(spec: str):
    """Spec: 'random'."""
    if spec != "random":
        raise ValueError("unknown random spec")
    return lambda pid, rng: UniformRandomBot(pid, rng)


def make_alwaysfold_factory(spec: str):
    if spec != "alwaysfold":
        raise ValueError("unknown alwaysfold spec")
    return lambda pid, rng: AlwaysFoldBot(pid, rng)


def make_openspiel_factory(model_path: str):
    """Load the MCCFR solver once and share it across both seats."""
    solver = load_solver(model_path)
    return lambda pid, rng: OpenSpielMCCFRBot(pid, solver, rng)


def make_lucy_factory(binary: str, model: str, abstraction: str = "fcpa",
                      hand_abstraction: str = "v1"):
    return lambda pid, rng: LucyBot(pid, binary, model, abstraction=abstraction,
                                    hand_abstraction=hand_abstraction,
                                    rng=rng)


def make_factory(spec: str):
    """Resolve a CLI bot spec like 'lucy:bin=...,model=...' into a factory."""
    if spec == "random":
        return make_random_factory(spec)
    if spec == "alwaysfold":
        return make_alwaysfold_factory(spec)
    if spec.startswith("openspiel:"):
        path = spec[len("openspiel:"):]
        return make_openspiel_factory(path)
    if spec.startswith("lucy:"):
        kvs = dict(kv.split("=", 1) for kv in spec[len("lucy:"):].split(","))
        return make_lucy_factory(
            binary=kvs["bin"],
            model=kvs["model"],
            abstraction=kvs.get("abstraction", "fcpa"),
            hand_abstraction=kvs.get("hand_abstraction", "v1"),
        )
    raise ValueError(f"unknown bot spec: {spec}")


# ---------------------------------------------------------------------------
# Hand player. One hand of HU NLHE FCPA, returning per-player utility.
#
# We do *not* use OpenSpiel's evaluate_bots() because we want per-pair seat
# swapping with identical decks. Instead we step the state ourselves, sampling
# chance nodes from a controlled RNG and player nodes via each bot.
# ---------------------------------------------------------------------------


def _play_one_hand(game, bots, chance_rng: np.random.Generator):
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            probs = np.asarray(probs, dtype=np.float64)
            # Defensive renormalization.
            probs = probs / probs.sum()
            a = int(chance_rng.choice(actions, p=probs))
            state.apply_action(a)
        else:
            cur = state.current_player()
            action = bots[cur].step(state)
            state.apply_action(action)
    return state.returns()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    bot_a: str
    bot_b: str
    num_pairs: int
    num_hands: int
    mbb_per_hand_a: float          # mean of paired diffs, in mbb/hand
    mbb_per_hand_a_se: float       # standard error of the mean
    ci95_low: float
    ci95_high: float
    raw_returns_a_seat0: List[float]
    raw_returns_a_seat1: List[float]
    seconds: float


def _ci95(values: np.ndarray) -> tuple[float, float, float]:
    """Returns (mean, std_error, half_width_95). Uses normal approx."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(values.mean())
    if n == 1:
        return mean, 0.0, 0.0
    se = float(values.std(ddof=1) / math.sqrt(n))
    return mean, se, 1.96 * se


# ---------------------------------------------------------------------------
# Main match loop
# ---------------------------------------------------------------------------

def play_match(
    factory_a: Callable, factory_b: Callable,
    num_pairs: int, seed: int,
    log_every: Optional[int] = None,
) -> MatchResult:
    """Play ``num_pairs`` duplicate pairs (= 2 * num_pairs hands).

    For each pair:
      1. Save chance-RNG state, play hand with seats [A=0, B=1].
      2. Restore chance-RNG state, play hand with seats [B=0, A=1].

    Each bot side gets a fresh stochastic-RNG seeded deterministically per
    pair so action sampling is reproducible.
    """
    game = pyspiel.load_game(HU_NLHE_FCPA_GAMEDEF)
    chance_rng = np.random.default_rng(seed)

    a_returns_seat0: list[float] = []  # A in seat 0 (BB) per pair
    a_returns_seat1: list[float] = []  # A in seat 1 (SB / button) per pair

    if log_every is None:
        log_every = max(1, num_pairs // 20)

    t0 = time.time()
    for pair in range(num_pairs):
        # --- Hand 1: A in seat 0, B in seat 1 ------------------------------
        chance_state = chance_rng.bit_generator.state
        a_rng = np.random.default_rng(seed + 1_000_000 * pair + 1)
        b_rng = np.random.default_rng(seed + 1_000_000 * pair + 2)
        bot_a0 = factory_a(0, a_rng)
        bot_b1 = factory_b(1, b_rng)
        try:
            ret = _play_one_hand(game, [bot_a0, bot_b1], chance_rng)
        finally:
            del bot_a0, bot_b1
        a_returns_seat0.append(float(ret[0]))

        # --- Hand 2: B in seat 0, A in seat 1, SAME deck -------------------
        chance_rng.bit_generator.state = chance_state
        a_rng = np.random.default_rng(seed + 1_000_000 * pair + 3)
        b_rng = np.random.default_rng(seed + 1_000_000 * pair + 4)
        bot_b0 = factory_b(0, b_rng)
        bot_a1 = factory_a(1, a_rng)
        try:
            ret = _play_one_hand(game, [bot_b0, bot_a1], chance_rng)
        finally:
            del bot_b0, bot_a1
        a_returns_seat1.append(float(ret[1]))

        if (pair + 1) % log_every == 0 or pair == 0:
            partial = (np.array(a_returns_seat0) + np.array(a_returns_seat1))
            mean = float(partial.mean()) / 2.0  # avg per hand within pair
            mbb = mean / BIG_BLIND * 1000.0
            print(f"[match] pair {pair + 1}/{num_pairs} "
                  f"A_mbb/hand≈{mbb:+.1f} "
                  f"elapsed={time.time() - t0:.1f}s", flush=True)

    seconds = time.time() - t0

    # Per-pair total return for A (sum of two hands), divide by 2 to get
    # per-hand. Then convert to mbb (chip BB == BIG_BLIND chips).
    per_pair_per_hand = (np.asarray(a_returns_seat0) +
                         np.asarray(a_returns_seat1)) / 2.0
    mbb_pair = per_pair_per_hand / BIG_BLIND * 1000.0
    mean, se, half = _ci95(mbb_pair)

    return MatchResult(
        bot_a="?",
        bot_b="?",
        num_pairs=num_pairs,
        num_hands=2 * num_pairs,
        mbb_per_hand_a=mean,
        mbb_per_hand_a_se=se,
        ci95_low=mean - half,
        ci95_high=mean + half,
        raw_returns_a_seat0=a_returns_seat0,
        raw_returns_a_seat1=a_returns_seat1,
        seconds=seconds,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bot-a", required=True,
                   help=("Bot spec, e.g. 'random', 'alwaysfold', "
                         "'openspiel:/path/to/model.pkl', "
                         "'lucy:bin=...,model=...,abstraction=fcpa'"))
    p.add_argument("--bot-b", required=True, help="Same spec format as --bot-a")
    p.add_argument("--label-a", default="A", help="Display label for bot A")
    p.add_argument("--label-b", default="B", help="Display label for bot B")
    p.add_argument("--pairs", type=int, default=500,
                   help="Number of duplicate pairs (= 2x hands).")
    p.add_argument("--seed", type=int, default=12345,
                   help="Chance-RNG seed for the match.")
    p.add_argument("--out", help="Write JSON result to this path.")
    args = p.parse_args()

    fa = make_factory(args.bot_a)
    fb = make_factory(args.bot_b)

    print(f"[match] {args.label_a} vs {args.label_b}, "
          f"pairs={args.pairs} seed={args.seed}", flush=True)
    result = play_match(fa, fb, args.pairs, args.seed)
    result.bot_a = args.label_a
    result.bot_b = args.label_b

    print(f"[match] {args.label_a} mbb/hand = "
          f"{result.mbb_per_hand_a:+.1f} ± {1.96 * result.mbb_per_hand_a_se:.1f} "
          f"(95% CI [{result.ci95_low:+.1f}, {result.ci95_high:+.1f}], "
          f"n={result.num_hands} hands, {result.seconds:.1f}s)",
          flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            d = asdict(result)
            json.dump(d, f, indent=2)
        print(f"[match] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
