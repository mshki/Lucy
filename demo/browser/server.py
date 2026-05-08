from __future__ import annotations

import os
import queue
import sys
import threading
import time
from typing import Optional

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402

import pyspiel  # noqa: E402

from bench.common import HU_NLHE_FCPA_GAMEDEF, BIG_BLIND, card_id_to_str  # noqa: E402
from bench.lucy_bot import LucyBot  # noqa: E402
from bench.openspiel_bot import OpenSpielMCCFRBot, load_solver  # noqa: E402


ACTION_NAME = {0: "fold", 1: "check/call", 2: "pot raise", 3: "all-in"}


class HumanBot(pyspiel.Bot):
    def __init__(self, player_id: int, action_queue: queue.Queue):
        super().__init__()
        self._player_id = player_id
        self._queue = action_queue

    def player_id(self) -> int:
        return self._player_id

    def restart_at(self, state):
        return

    def step(self, state) -> int:
        legal = state.legal_actions(self._player_id)
        if not legal:
            return pyspiel.INVALID_ACTION
        while True:
            try:
                action = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if action == -1:
                return legal[0]
            if action in legal:
                return action


def _hole_cards_for(state, player_id: int) -> list[str]:
    full = state.full_history()
    chance_actions = [pa.action for pa in full
                      if pa.player == pyspiel.PlayerId.CHANCE]
    holes = [[] for _ in range(2)]
    for i, c in enumerate(chance_actions[:4]):
        holes[i % 2].append(c)
    return [card_id_to_str(c) for c in holes[player_id]]


def _board_cards(state) -> list[str]:
    full = state.full_history()
    chance_actions = [pa.action for pa in full
                      if pa.player == pyspiel.PlayerId.CHANCE]
    return [card_id_to_str(c) for c in chance_actions[4:]]


def _bets_per_player(state) -> tuple[int, int]:
    s = str(state)
    p0 = p1 = 0
    for line in s.splitlines():
        line = line.strip()
        if line.startswith("P0 Money:"):
            try:
                p0 = 200 - int(line.split(":")[1].strip())
            except Exception:
                pass
        elif line.startswith("P1 Money:"):
            try:
                p1 = 200 - int(line.split(":")[1].strip())
            except Exception:
                pass
    return p0, p1


class SpectatorMatch:
    def __init__(self, label_a: str, label_b: str,
                 factory_a, factory_b,
                 action_delay_s: float = 1.4,
                 hand_delay_s: float = 2.5,
                 starting_stack_chips: int = 200,
                 human_seat: Optional[int] = None,
                 human_queue: Optional[queue.Queue] = None):
        self.label_a = label_a
        self.label_b = label_b
        self.factory_a = factory_a
        self.factory_b = factory_b
        self.action_delay_s = action_delay_s
        self.hand_delay_s = hand_delay_s
        self.starting_stack_chips = starting_stack_chips
        self.human_seat = human_seat
        self.human_queue = human_queue

        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()

        self._snapshot = self._initial_snapshot()

    def _initial_snapshot(self) -> dict:
        return {
            "running": False,
            "mode": "human" if self.human_seat is not None else "spectator",
            "label_a": self.label_a,
            "label_b": self.label_b,
            "chips_a_total": 0,
            "chips_b_total": 0,
            "hand_count": 0,
            "current_hand_actor": None,
            "stage": "ready",
            "hole_a": [],
            "hole_b": [],
            "board": [],
            "pot": 0,
            "bet_a": 0,
            "bet_b": 0,
            "log": [],
            "last_action": None,
            "winner_text": None,
            "human_seat": self.human_seat,
            "human_turn": False,
            "legal_actions": [],
        }

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self._snapshot)

    def _set(self, **kwargs):
        with self.lock:
            self._snapshot.update(kwargs)

    def _log(self, msg: str):
        with self.lock:
            log = list(self._snapshot.get("log", []))
            log.append(msg)
            self._snapshot["log"] = log[-40:]

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_flag.clear()
        self._set(running=True, log=[], hand_count=0,
                  chips_a_total=0, chips_b_total=0)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag.set()
        self._set(running=False)

    def _run(self):
        game = pyspiel.load_game_as_turn_based(HU_NLHE_FCPA_GAMEDEF)
        chips_a_total = 0
        chips_b_total = 0
        hand_count = 0

        while not self.stop_flag.is_set():
            hand_count += 1
            seed = 17 * hand_count + 31415
            rng = np.random.default_rng(seed)
            if self.human_seat is None:
                seat_a = (hand_count - 1) % 2
                seat_b = 1 - seat_a
            else:
                seat_b = self.human_seat
                seat_a = 1 - seat_b

            try:
                bot_a = self.factory_a(seat_a, rng)
                bot_b = self.factory_b(seat_b, rng)
            except Exception as e:
                self._log(f"bot factory error: {e}")
                self._set(running=False)
                return
            bots = [None, None]
            bots[seat_a] = bot_a
            bots[seat_b] = bot_b
            seat_label = ["", ""]
            seat_label[seat_a] = self.label_a
            seat_label[seat_b] = self.label_b

            self._set(stage="dealing", hand_count=hand_count,
                      hole_a=[], hole_b=[], board=[],
                      pot=3, bet_a=(2 if seat_a == 0 else 1),
                      bet_b=(2 if seat_b == 0 else 1),
                      last_action=None, winner_text=None,
                      seat_a=seat_a, seat_b=seat_b)

            r0, r1 = self._play_one_hand(game, bots, seat_label)

            chips_a_total += r0 if seat_a == 0 else r1
            chips_b_total += r0 if seat_b == 0 else r1

            self._set(chips_a_total=chips_a_total,
                      chips_b_total=chips_b_total)

            if not self.stop_flag.is_set():
                time.sleep(self.hand_delay_s)

            try:
                if hasattr(bot_a, "_proc"):
                    bot_a._proc.terminate()
                if hasattr(bot_b, "_proc"):
                    bot_b._proc.terminate()
            except Exception:
                pass

        self._set(running=False, stage="stopped")

    def _play_one_hand(self, game, bots, seat_label):
        state = game.new_initial_state()
        chance_rng = np.random.default_rng()

        while not state.is_terminal():
            if self.stop_flag.is_set():
                return 0, 0

            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                probs = np.asarray(probs, dtype=np.float64)
                probs = probs / probs.sum()
                a = int(chance_rng.choice(actions, p=probs))
                state.apply_action(a)
                self._refresh_snapshot(state, seat_label)
                continue

            cur = state.current_player()
            is_human_turn = (self.human_seat is not None
                              and cur == self.human_seat)
            legal = list(state.legal_actions(cur)) if is_human_turn else []
            self._refresh_snapshot(state, seat_label, current_actor=cur,
                                    is_human_turn=is_human_turn,
                                    legal_actions=legal)
            if not is_human_turn and not self.stop_flag.is_set():
                time.sleep(self.action_delay_s)

            try:
                action = bots[cur].step(state)
            except Exception as e:
                self._log(f"bot {seat_label[cur]} error: {e}")
                return 0, 0
            if is_human_turn:
                self._set(human_turn=False, legal_actions=[])

            self._set(last_action={
                "player": cur,
                "label": seat_label[cur],
                "action": ACTION_NAME.get(action, str(action)),
            })
            self._log(f"hand {self._snapshot['hand_count']}  "
                      f"{seat_label[cur]:>22}  {ACTION_NAME.get(action, action)}")

            state.apply_action(action)
            self._refresh_snapshot(state, seat_label)

        returns = state.returns()
        winner_text = self._winner_text(state, seat_label, returns)
        self._set(winner_text=winner_text, current_hand_actor=None,
                  stage="hand_done")
        self._log(winner_text)
        self._refresh_snapshot(state, seat_label, reveal_all=True)
        return float(returns[0]), float(returns[1])

    def _winner_text(self, state, seat_label, returns) -> str:
        r0, r1 = returns
        if r0 > 0:
            return f"hand to {seat_label[0]}  ({r0:+.0f} chips)"
        if r1 > 0:
            return f"hand to {seat_label[1]}  ({r1:+.0f} chips)"
        return "hand split"

    def _refresh_snapshot(self, state, seat_label,
                           current_actor: Optional[int] = None,
                           reveal_all: bool = False,
                           is_human_turn: bool = False,
                           legal_actions: Optional[list] = None):
        hole0 = _hole_cards_for(state, 0)
        hole1 = _hole_cards_for(state, 1)
        board = _board_cards(state)
        bet0, bet1 = _bets_per_player(state)
        pot = bet0 + bet1

        # In human-vs-bot mode, hide the bot's hole cards until showdown.
        hole_a_show = hole0 if seat_label[0] == self.label_a else hole1
        hole_b_show = hole0 if seat_label[0] == self.label_b else hole1
        if self.human_seat is not None and not reveal_all:
            human_label = (self.label_a if self.human_seat == 0
                                       else self.label_b)
            if seat_label[self.human_seat] == self.label_a:
                hole_b_show = []
            else:
                hole_a_show = []

        self._set(
            hole_a=hole_a_show,
            hole_b=hole_b_show,
            board=board,
            pot=pot,
            bet_a=bet0 if seat_label[0] == self.label_a else bet1,
            bet_b=bet0 if seat_label[0] == self.label_b else bet1,
            current_hand_actor=(seat_label[current_actor]
                                 if current_actor is not None else None),
            human_turn=is_human_turn,
            legal_actions=legal_actions or [],
            stage=("preflop" if len(board) == 0 else
                   "flop" if len(board) == 3 else
                   "turn" if len(board) == 4 else "river"),
        )


def make_lucy_factory(binary: str, model: str,
                      hand_abstraction: str = "v3ir"):
    def factory(player_id: int, rng: np.random.Generator):
        return LucyBot(player_id, binary, model,
                       abstraction="fcpa",
                       hand_abstraction=hand_abstraction,
                       rng=rng)
    return factory


def make_openspiel_factory(model_path: str):
    solver = load_solver(model_path)
    def factory(player_id: int, rng: np.random.Generator):
        return OpenSpielMCCFRBot(player_id, solver, rng)
    return factory


def make_human_factory(action_queue: queue.Queue):
    def factory(player_id: int, rng: np.random.Generator):
        return HumanBot(player_id, action_queue)
    return factory


app = Flask(__name__,
            static_folder=os.path.dirname(os.path.abspath(__file__)))
MATCH: Optional[SpectatorMatch] = None
HUMAN_QUEUE: Optional[queue.Queue] = None


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/state")
def api_state():
    if MATCH is None:
        return jsonify({"running": False,
                        "log": ["match not configured at server start"]})
    return jsonify(MATCH.snapshot())


@app.route("/api/start", methods=["POST"])
def api_start():
    if MATCH is None:
        return jsonify({"error": "match not configured"}), 400
    MATCH.start()
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    if MATCH is None:
        return jsonify({"error": "match not configured"}), 400
    MATCH.stop()
    return jsonify({"ok": True})


@app.route("/api/human/act", methods=["POST"])
def api_human_act():
    if HUMAN_QUEUE is None:
        return jsonify({"error": "human mode not enabled"}), 400
    data = request.get_json(silent=True) or {}
    try:
        action = int(data.get("action", -1))
    except (TypeError, ValueError):
        return jsonify({"error": "bad action"}), 400
    if action not in (0, 1, 2, 3):
        return jsonify({"error": "action must be 0..3"}), 400
    HUMAN_QUEUE.put(action)
    return jsonify({"ok": True})


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--lucy-binary", required=True)
    p.add_argument("--lucy-model", required=True)
    p.add_argument("--lucy-hand-abstraction", default="v3ir")
    p.add_argument("--lucy-label", default="Lucy GPU V3")
    p.add_argument("--opp-kind", default="openspiel",
                   choices=["openspiel", "lucy", "human"])
    p.add_argument("--opp-model", default="")
    p.add_argument("--opp-hand-abstraction", default="v1ir")
    p.add_argument("--opp-label", default="OpenSpiel @ 1M iters")
    p.add_argument("--human-label", default="You")
    p.add_argument("--port", type=int, default=5050)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--action-delay", type=float, default=1.4)
    p.add_argument("--hand-delay", type=float, default=2.5)
    args = p.parse_args()

    global MATCH, HUMAN_QUEUE

    if args.opp_kind == "human":
        HUMAN_QUEUE = queue.Queue()
        factory_a = make_lucy_factory(args.lucy_binary, args.lucy_model,
                                       args.lucy_hand_abstraction)
        factory_b = make_human_factory(HUMAN_QUEUE)
        MATCH = SpectatorMatch(
            label_a=args.lucy_label,
            label_b=args.human_label,
            factory_a=factory_a,
            factory_b=factory_b,
            action_delay_s=args.action_delay,
            hand_delay_s=args.hand_delay,
            human_seat=1,
            human_queue=HUMAN_QUEUE,
        )
        print(f"[demo] {args.lucy_label} vs {args.human_label} (human)",
              file=sys.stderr)
    else:
        if not args.opp_model:
            print("--opp-model required for non-human matchups", file=sys.stderr)
            sys.exit(1)
        factory_a = make_lucy_factory(args.lucy_binary, args.lucy_model,
                                       args.lucy_hand_abstraction)
        if args.opp_kind == "openspiel":
            factory_b = make_openspiel_factory(args.opp_model)
        else:
            factory_b = make_lucy_factory(args.lucy_binary, args.opp_model,
                                           args.opp_hand_abstraction)
        MATCH = SpectatorMatch(
            label_a=args.lucy_label,
            label_b=args.opp_label,
            factory_a=factory_a,
            factory_b=factory_b,
            action_delay_s=args.action_delay,
            hand_delay_s=args.hand_delay,
        )
        print(f"[demo] {args.lucy_label} vs {args.opp_label}", file=sys.stderr)

    print(f"[demo] http://{args.host}:{args.port}", file=sys.stderr)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
