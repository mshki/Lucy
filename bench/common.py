"""Shared constants and helpers for the Lucy benchmark harness.

The benchmark plays Heads-Up No-Limit Texas Hold'em using OpenSpiel's
``universal_poker`` game with the FCPA (Fold/Call/Pot/All-in) betting
abstraction. Both Lucy variants and the OpenSpiel baseline solver train
and play in this exact game.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Game definition
# ---------------------------------------------------------------------------

# Heads-up No-Limit Texas Hold'em with FCPA betting abstraction. Stacks 100bb,
# blinds 1/2. Action IDs in this abstraction are stable:
#   0 = FOLD, 1 = CHECK/CALL, 2 = POT-sized bet/raise, 3 = ALL-IN
HU_NLHE_FCPA_GAMEDEF = (
    "universal_poker("
    "betting=nolimit,"
    "numPlayers=2,"
    "numRounds=4,"
    "blind=2 1,"
    "firstPlayer=2 1 1 1,"
    "numSuits=4,"
    "numRanks=13,"
    "numHoleCards=2,"
    "numBoardCards=0 3 1 1,"
    "stack=200 200,"
    "bettingAbstraction=fcpa"
    ")"
)

# Match these values with Lucy's CLI flags so both bots train on the same game.
NUM_PLAYERS = 2
SMALL_BLIND = 1.0
BIG_BLIND = 2.0
STARTING_STACK = 200.0  # 100bb in chip units
STACK_BB = 100.0

# OpenSpiel HU NLHE: in `blind=2 1`, player 0 posts 2 (BB), player 1 posts 1
# (SB / Button). Player 1 acts first preflop; player 0 acts first postflop.
# Lucy's HU convention with dealer=0: SB sits at (0+1)%2 = 1, BB at 0. So Lucy
# player 0 == OpenSpiel player 0 == BB, and Lucy player 1 == OpenSpiel player 1
# == SB. We pass dealer=0 in every Lucy serve query.
LUCY_DEALER = 0


# ---------------------------------------------------------------------------
# FCPA action id → human-readable label (for logs)
# ---------------------------------------------------------------------------

ACTION_LABEL = {0: "F", 1: "C", 2: "P", 3: "A"}


# ---------------------------------------------------------------------------
# Card encoding
# ---------------------------------------------------------------------------
#
# OpenSpiel universal_poker indexes cards as ``rank * numSuits + suit`` where
# numSuits=4 and numRanks=13. Rank goes 2,3,...,K,A → 0,1,...,11,12. Suit goes
# clubs=0, diamonds=1, hearts=2, spades=3. This matches Lucy's
# ``Card(rank=Rank::TWO..ACE, suit=Suit::CLUBS..SPADES)`` exactly.

_RANK_CHARS = "23456789TJQKA"
_SUIT_CHARS = "cdhs"


def card_id_to_str(card_id: int) -> str:
    """Convert a 0..51 card index to a 'Ah'-style two-char card string."""
    rank = card_id // 4
    suit = card_id % 4
    return f"{_RANK_CHARS[rank]}{_SUIT_CHARS[suit]}"


def card_str_to_id(card: str) -> int:
    rank = _RANK_CHARS.index(card[0].upper())
    # Lucy and the JSON protocol use lowercase suits.
    suit = _SUIT_CHARS.index(card[1].lower())
    return rank * 4 + suit


# ---------------------------------------------------------------------------
# State extraction (OpenSpiel universal_poker → fields needed by Lucy serve)
# ---------------------------------------------------------------------------


@dataclass
class HandSnapshot:
    """A snapshot of the hand state from an OpenSpiel universal_poker state.

    Fields are the inputs Lucy's serve mode needs to rebuild its own state and
    look up a strategy.
    """
    hero: int                     # player whose turn it is
    hole: list[str]               # 2 cards for ``hero`` (e.g. ['Ah','Kd'])
    board: list[str]              # 0/3/4/5 cards depending on stage
    stage: str                    # 'preflop'|'flop'|'turn'|'river'
    history: list[dict]           # [{"player": p, "action": a}, ...] FCPA ids
    legal_actions: list[int]      # FCPA ids legal in this state


def _stage_from_board_size(n: int) -> str:
    if n == 0:
        return "preflop"
    if n == 3:
        return "flop"
    if n == 4:
        return "turn"
    if n == 5:
        return "river"
    raise ValueError(f"unexpected board size {n}")


def snapshot_from_openspiel(state) -> HandSnapshot:
    """Extract a HandSnapshot from an OpenSpiel universal_poker state.

    Walk ``state.full_history()`` and split chance-node card-deals from the
    player's FCPA actions. Cards are dealt in OpenSpiel's standard order:
        1) one hole card per player (alternating), repeated until each player
           has ``numHoleCards`` cards;
        2) flop cards (3), then turn (1), then river (1) at the appropriate
           street boundaries.
    """
    full = state.full_history()
    num_players = NUM_PLAYERS
    num_hole = 2

    # `full_history` returns a list of pyspiel.PlayerAction objects with
    # `.player` and `.action`. Chance moves have `player == pyspiel.PlayerId.CHANCE`.
    import pyspiel  # imported lazily to avoid hard requirement at module import
    chance_actions = [pa.action for pa in full if pa.player == pyspiel.PlayerId.CHANCE]
    player_actions = [(pa.player, pa.action) for pa in full
                      if pa.player != pyspiel.PlayerId.CHANCE]

    # Hole cards: first num_hole * num_players chance actions, alternating.
    holes_each: list[list[int]] = [[] for _ in range(num_players)]
    consumed = 0
    for i in range(num_hole * num_players):
        if consumed >= len(chance_actions):
            break
        # Alternating: position-i goes to player i % num_players.
        holes_each[i % num_players].append(chance_actions[consumed])
        consumed += 1
    board_cards = chance_actions[consumed:]

    hero = state.current_player()
    hole_strs = [card_id_to_str(c) for c in holes_each[hero]]
    board_strs = [card_id_to_str(c) for c in board_cards]
    legal = list(state.legal_actions(hero))
    history = [{"player": int(p), "action": int(a)} for (p, a) in player_actions]

    return HandSnapshot(
        hero=hero,
        hole=hole_strs,
        board=board_strs,
        stage=_stage_from_board_size(len(board_strs)),
        history=history,
        legal_actions=legal,
    )


# ---------------------------------------------------------------------------
# File layout helpers (standard project paths used by SLURM scripts).
# ---------------------------------------------------------------------------

def workspace_root() -> str:
    """Root for benchmark artifacts — env override or default."""
    return os.environ.get("LUCY_BENCH_ROOT", os.getcwd())


def models_dir() -> str:
    return os.path.join(workspace_root(), "models")


def results_dir() -> str:
    return os.path.join(workspace_root(), "results")


def logs_dir() -> str:
    return os.path.join(workspace_root(), "logs")


def model_path(solver: str, iters: int, seed: int) -> str:
    """Canonical path for a saved model, e.g. models/lucy-cuda/iters=10000/seed=1.dat"""
    return os.path.join(models_dir(), solver, f"iters={iters}", f"seed={seed}.dat")


def result_path(matchup: str, lucy_iters: int, baseline_iters: int,
                seed: int, ext: str = "json") -> str:
    return os.path.join(
        results_dir(), matchup,
        f"lucy_iters={lucy_iters}",
        f"baseline_iters={baseline_iters}",
        f"seed={seed}.{ext}",
    )
