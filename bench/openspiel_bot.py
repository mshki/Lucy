"""Bots and policy I/O for the OpenSpiel side of the benchmark."""
from __future__ import annotations

import os
import pickle
from typing import Dict, Iterable

import numpy as np
import pyspiel  # type: ignore

from .common import HU_NLHE_FCPA_GAMEDEF


# ---------------------------------------------------------------------------
# A serializable snapshot of an OpenSpiel MCCFR average policy.
#
# The MCCFR solver itself is C++ and pickling it can break across OpenSpiel
# versions. We extract a plain Python dict from
# ``solver.average_policy().action_probabilities(state)`` for every reachable
# information state would be intractable for full NLHE — instead we keep the
# policy *callable* via the in-memory solver during evaluation, and use pickle
# only as a "hot reload" mechanism that round-trips the C++ object across
# sub-processes (tested as supported in the OpenSpiel research notes).
# ---------------------------------------------------------------------------

def _load_game():
    return pyspiel.load_game(HU_NLHE_FCPA_GAMEDEF)


def save_solver(solver, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(solver, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_solver(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# pyspiel.Bot wrapping an MCCFR average policy.
# ---------------------------------------------------------------------------

class OpenSpielMCCFRBot(pyspiel.Bot):
    """Bot that samples actions from an OpenSpiel MCCFR average policy."""

    def __init__(self, player_id: int, solver, rng: np.random.Generator):
        super().__init__()
        self._player_id = player_id
        self._policy = solver.average_policy()
        self._rng = rng

    def player_id(self) -> int:
        return self._player_id

    def restart_at(self, state):  # noqa: D401 — pyspiel signature
        return

    def step(self, state) -> int:
        legal = state.legal_actions(self._player_id)
        if not legal:
            return pyspiel.INVALID_ACTION
        probs_dict: Dict[int, float] = self._policy.action_probabilities(state)
        # Restrict to legal and renormalize.
        items = [(a, probs_dict.get(a, 0.0)) for a in legal]
        actions = [a for a, _ in items]
        weights = np.asarray([w for _, w in items], dtype=np.float64)
        s = float(weights.sum())
        if s <= 0.0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / s
        return int(self._rng.choice(actions, p=weights))


# ---------------------------------------------------------------------------
# Cheap baselines (sanity floors).
# ---------------------------------------------------------------------------

class UniformRandomBot(pyspiel.Bot):
    """Uniform-random over legal actions. Used as a sanity floor."""

    def __init__(self, player_id: int, rng: np.random.Generator):
        super().__init__()
        self._player_id = player_id
        self._rng = rng

    def player_id(self) -> int:
        return self._player_id

    def restart_at(self, state):
        return

    def step(self, state) -> int:
        legal = state.legal_actions(self._player_id)
        return int(self._rng.choice(legal)) if legal else pyspiel.INVALID_ACTION


class AlwaysFoldBot(pyspiel.Bot):
    """Folds whenever fold is legal, else check/calls. Floor baseline."""

    def __init__(self, player_id: int, rng: np.random.Generator):
        super().__init__()
        self._player_id = player_id
        self._rng = rng

    def player_id(self) -> int:
        return self._player_id

    def restart_at(self, state):
        return

    def step(self, state) -> int:
        legal = state.legal_actions(self._player_id)
        if 0 in legal:
            return 0
        if 1 in legal:
            return 1
        return int(self._rng.choice(legal)) if legal else pyspiel.INVALID_ACTION
