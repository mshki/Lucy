"""``pyspiel.Bot`` wrapper that delegates strategy queries to a Lucy subprocess.

Each Lucy process is started once at construction with ``--serve <model>`` and
stays alive across all hands; queries are line-delimited JSON over stdin/stdout
(see ``serve_mode`` in ``src/main.cpp``). The wrapper is responsible for
translating each OpenSpiel state into the JSON request shape Lucy expects, and
sampling an action from Lucy's returned ``legal_actions`` / ``probabilities``.

Sampling is done by the wrapper (with its own RNG) rather than trusting Lucy's
returned ``action`` field — this gives the harness reproducibility independent
of Lucy's PRNG.
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Optional

import numpy as np
import pyspiel  # type: ignore

from .common import (
    BIG_BLIND, LUCY_DEALER, NUM_PLAYERS, SMALL_BLIND, STARTING_STACK,
    snapshot_from_openspiel,
)


class LucyBot(pyspiel.Bot):
    """Plays Heads-Up NLHE FCPA via a long-running Lucy subprocess."""

    def __init__(self, player_id: int, lucy_binary: str, model_path: str,
                 abstraction: str = "fcpa", rng: Optional[np.random.Generator] = None,
                 stderr_to_devnull: bool = True):
        super().__init__()
        if not os.path.exists(lucy_binary):
            raise FileNotFoundError(f"Lucy binary not found: {lucy_binary}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Lucy model not found: {model_path}")
        self._player_id = player_id
        self._abstraction = abstraction
        self._rng = rng or np.random.default_rng()
        self._proc = subprocess.Popen(
            [lucy_binary, "--serve", model_path, "--abstraction", abstraction],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL if stderr_to_devnull else None,
            text=True,
            bufsize=1,
        )

    # -- pyspiel.Bot API --------------------------------------------------

    def player_id(self) -> int:
        return self._player_id

    def restart_at(self, state):  # noqa: D401 - pyspiel signature
        # Lucy is stateless across queries (each request carries the full
        # state). Nothing to restart.
        return

    def step(self, state) -> int:
        legal_set = set(state.legal_actions(self._player_id))
        if not legal_set:
            return pyspiel.INVALID_ACTION

        snap = snapshot_from_openspiel(state)
        req = {
            "player_id": self._player_id,
            "num_players": NUM_PLAYERS,
            "dealer": LUCY_DEALER,
            "sb": SMALL_BLIND,
            "bb": BIG_BLIND,
            "starting_stack": STARTING_STACK,
            "stage": snap.stage,
            "hole": snap.hole,
            "board": snap.board,
            "history": snap.history,
            "abstraction": self._abstraction,
        }
        line = json.dumps(req)
        assert self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()
        resp_line = self._proc.stdout.readline()
        if not resp_line:
            raise RuntimeError("Lucy subprocess closed unexpectedly")
        try:
            resp = json.loads(resp_line)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Lucy returned non-JSON: {resp_line!r}") from e
        if not resp.get("ok"):
            raise RuntimeError(f"Lucy error: {resp.get('error')}")

        legal_ids = resp["legal_actions"]
        probs = resp["probabilities"]
        if len(legal_ids) != len(probs):
            raise RuntimeError(
                f"Lucy legal/probs mismatch: {legal_ids} vs {probs}"
            )

        # Restrict to OpenSpiel-legal subset (defensive: Lucy and OpenSpiel
        # should agree under FCPA, but bet sizing in pot mode can occasionally
        # become illegal at the boundary, e.g. when the pot-raise would leave
        # < min-raise chips for the opponent).
        filtered = [(a, p) for a, p in zip(legal_ids, probs) if a in legal_set]
        if not filtered:
            # Fall back to uniform over OpenSpiel-legal actions.
            return int(self._rng.choice(sorted(legal_set)))
        actions, weights = zip(*filtered)
        weights = np.asarray(weights, dtype=np.float64)
        s = weights.sum()
        if s <= 0.0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / s
        return int(self._rng.choice(actions, p=weights))

    def __del__(self):
        try:
            if hasattr(self, "_proc") and self._proc.poll() is None:
                self._proc.stdin.close()  # type: ignore[union-attr]
                self._proc.wait(timeout=2)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
