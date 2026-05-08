from __future__ import annotations

import json
import os
import random
import secrets
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    from flask import Flask, request, jsonify, send_from_directory
except ImportError:
    print("[demo] missing flask. Install with: pip install flask", file=sys.stderr)
    sys.exit(1)


RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card_str(c: int) -> str:
    return RANKS[c // 4] + SUITS[c % 4]


def card_int(s: str) -> int:
    return RANKS.index(s[0]) * 4 + SUITS.index(s[1])


def eval_7card(cards: list[int]) -> int:
    n = len(cards)
    rank_count = [0] * 13
    suit_mask = [0] * 4
    for c in cards:
        rank_count[c >> 2] += 1
        suit_mask[c & 3] |= 1 << (c >> 2)

    flush_suit = -1
    for s in range(4):
        if bin(suit_mask[s]).count("1") >= 5:
            flush_suit = s
            break

    any_rank = 0
    for r in range(13):
        if rank_count[r] > 0:
            any_rank |= 1 << r

    top_straight = -1
    for top in range(12, 3, -1):
        window = 0x1F << (top - 4)
        if (any_rank & window) == window:
            top_straight = top
            break
    if top_straight < 0 and (any_rank & 0x100F) == 0x100F:
        top_straight = 3

    top_sflush = -1
    if flush_suit >= 0:
        fmask = suit_mask[flush_suit]
        for top in range(12, 3, -1):
            window = 0x1F << (top - 4)
            if (fmask & window) == window:
                top_sflush = top
                break
        if top_sflush < 0 and (fmask & 0x100F) == 0x100F:
            top_sflush = 3

    def pack(cat, r0=0, r1=0, r2=0, r3=0, r4=0):
        return ((cat & 0xF) << 24) | \
               ((r0 & 0xF) << 20) | ((r1 & 0xF) << 16) | \
               ((r2 & 0xF) << 12) | ((r3 & 0xF) << 8)  | ((r4 & 0xF) << 4)

    if top_sflush >= 0:
        return pack(9, top_sflush)

    quads = -1; trips1 = -1; trips2 = -1
    pair1 = -1; pair2 = -1
    for r in range(12, -1, -1):
        c = rank_count[r]
        if c == 4 and quads < 0: quads = r
        elif c == 3:
            if trips1 < 0: trips1 = r
            elif trips2 < 0: trips2 = r
        elif c == 2:
            if pair1 < 0: pair1 = r
            elif pair2 < 0: pair2 = r

    if quads >= 0:
        kicker = -1
        for r in range(12, -1, -1):
            if r != quads and rank_count[r] >= 1:
                kicker = r; break
        return pack(8, quads, kicker)

    if trips1 >= 0 and (trips2 >= 0 or pair1 >= 0):
        if trips2 >= 0 and (pair1 < 0 or trips2 > pair1):
            pair_rank = trips2
        else:
            pair_rank = pair1
        return pack(7, trips1, pair_rank)

    if flush_suit >= 0:
        fmask = suit_mask[flush_suit]
        rs = []
        for r in range(12, -1, -1):
            if fmask & (1 << r):
                rs.append(r)
                if len(rs) == 5: break
        rs += [0] * (5 - len(rs))
        return pack(6, *rs)

    if top_straight >= 0:
        return pack(5, top_straight)

    if trips1 >= 0:
        ks = []
        for r in range(12, -1, -1):
            if r != trips1 and rank_count[r] >= 1:
                ks.append(r)
                if len(ks) == 2: break
        ks += [0] * (2 - len(ks))
        return pack(4, trips1, ks[0], ks[1])

    if pair1 >= 0 and pair2 >= 0:
        kicker = 0
        for r in range(12, -1, -1):
            if r != pair1 and r != pair2 and rank_count[r] >= 1:
                kicker = r; break
        return pack(3, pair1, pair2, kicker)

    if pair1 >= 0:
        ks = []
        for r in range(12, -1, -1):
            if r != pair1 and rank_count[r] >= 1:
                ks.append(r)
                if len(ks) == 3: break
        ks += [0] * (3 - len(ks))
        return pack(2, pair1, ks[0], ks[1], ks[2])

    rs = []
    for r in range(12, -1, -1):
        if rank_count[r] >= 1:
            rs.append(r)
            if len(rs) == 5: break
    rs += [0] * (5 - len(rs))
    return pack(1, *rs)


@dataclass
class Player:
    seat: int
    name: str
    stack: int
    hole_cards: list[int] = field(default_factory=list)
    cur_bet: int = 0
    total_bet: int = 0
    folded: bool = False
    all_in: bool = False
    has_acted: bool = False


class PokerTable:
    def __init__(self, session_id: str, sb_chips: int = 1, bb_chips: int = 2,
                 starting_stack: int = 200):
        self.session_id = session_id
        self.lock = threading.Lock()
        self.players: dict[int, Player] = {}
        self.dealer_seat = 0
        self.sb_chips = sb_chips
        self.bb_chips = bb_chips
        self.starting_stack = starting_stack
        self.deck: list[int] = []
        self.board: list[int] = []
        self.pot = 0
        self.high_bet = 0
        self.stage = "waiting"
        self.current_seat: Optional[int] = None
        self.last_winner_info: Optional[str] = None
        self.log: list[str] = []
        self.coach_advice: dict = {}

    def join(self, name: str) -> int:
        with self.lock:
            seat = next((s for s in range(6) if s not in self.players), -1)
            if seat < 0:
                raise ValueError("Table full (6 seats max).")
            self.players[seat] = Player(seat=seat, name=name,
                                         stack=self.starting_stack)
            self.log.append(f"{name} joined seat {seat}.")
            return seat

    def leave(self, seat: int):
        with self.lock:
            if seat in self.players:
                p = self.players.pop(seat)
                self.log.append(f"{p.name} left.")

    def num_seated(self) -> int:
        return len(self.players)

    def start_hand(self):
        with self.lock:
            seats = sorted(s for s, p in self.players.items() if p.stack > 0)
            if len(seats) < 2:
                self.stage = "waiting"
                return
            for p in self.players.values():
                p.hole_cards = []
                p.cur_bet = 0
                p.total_bet = 0
                p.folded = False
                p.all_in = False
                p.has_acted = False
            self.board = []
            self.pot = 0
            self.high_bet = 0
            self.last_winner_info = None
            self.coach_advice = {}

            self.deck = list(range(52))
            random.shuffle(self.deck)

            for _ in range(2):
                for s in seats:
                    self.players[s].hole_cards.append(self.deck.pop())

            self.dealer_seat = seats[(seats.index(self.dealer_seat) + 1) %
                                     len(seats)] if self.dealer_seat in seats else seats[0]

            sb_seat = seats[(seats.index(self.dealer_seat) + 1) % len(seats)]
            bb_seat = seats[(seats.index(self.dealer_seat) + 2) % len(seats)]
            if len(seats) == 2:
                sb_seat = self.dealer_seat
                bb_seat = seats[(seats.index(self.dealer_seat) + 1) % 2]
            self._take_chips(self.players[sb_seat], self.sb_chips)
            self._take_chips(self.players[bb_seat], self.bb_chips)
            self.high_bet = self.bb_chips
            self.players[sb_seat].cur_bet = self.sb_chips
            self.players[sb_seat].total_bet = self.sb_chips
            self.players[bb_seat].cur_bet = self.bb_chips
            self.players[bb_seat].total_bet = self.bb_chips

            if len(seats) == 2:
                self.current_seat = self.dealer_seat
            else:
                self.current_seat = seats[(seats.index(bb_seat) + 1) %
                                          len(seats)]

            self.stage = "preflop"
            self.log.append(
                f"--- Hand started — dealer={self.players[self.dealer_seat].name} "
                f"SB={self.players[sb_seat].name} BB={self.players[bb_seat].name} ---"
            )

    def _take_chips(self, p: Player, amount: int) -> int:
        actual = min(amount, p.stack)
        p.stack -= actual
        p.total_bet += actual
        self.pot += actual
        if p.stack <= 0:
            p.all_in = True
            p.stack = 0
        return actual

    def act(self, seat: int, action: str, amount: int = 0):
        with self.lock:
            if self.current_seat != seat:
                raise ValueError(f"Not seat {seat}'s turn (current={self.current_seat}).")
            if self.stage in ("waiting", "done", "showdown"):
                raise ValueError(f"Cannot act in stage {self.stage}.")

            p = self.players[seat]
            if p.folded or p.all_in:
                raise ValueError("Player folded or all-in already.")

            if action == "fold":
                p.folded = True
                p.has_acted = True
                self.log.append(f"{p.name} folds.")

            elif action == "check":
                if p.cur_bet < self.high_bet:
                    raise ValueError("Cannot check; must call or raise.")
                p.has_acted = True
                self.log.append(f"{p.name} checks.")

            elif action == "call":
                call_amt = self.high_bet - p.cur_bet
                if call_amt <= 0:
                    raise ValueError("Nothing to call.")
                added = self._take_chips(p, call_amt)
                p.cur_bet += added
                p.has_acted = True
                self.log.append(f"{p.name} calls {added}.")

            elif action == "raise":
                add = amount - p.cur_bet
                if add <= 0 or amount <= self.high_bet:
                    raise ValueError(f"Raise must exceed high bet {self.high_bet}.")
                added = self._take_chips(p, add)
                p.cur_bet += added
                self.high_bet = p.cur_bet
                for sp in self.players.values():
                    if sp.seat != seat and not sp.folded and not sp.all_in:
                        sp.has_acted = False
                p.has_acted = True
                self.log.append(f"{p.name} raises to {p.cur_bet}.")

            else:
                raise ValueError(f"Unknown action: {action}")

            self._advance()

    def _advance(self):
        living = [p for p in self.players.values() if not p.folded]

        if len(living) == 1:
            self._end_hand_no_showdown(living[0])
            return

        round_closed = all(
            p.has_acted and (p.all_in or p.cur_bet == self.high_bet)
            for p in living
        )

        if not round_closed:
            self._advance_seat()
            return

        for p in self.players.values():
            p.cur_bet = 0
            p.has_acted = False
        self.high_bet = 0

        if self.stage == "preflop":
            self.stage = "flop"
            for _ in range(3):
                self.board.append(self.deck.pop())
            self.log.append(f"Flop: {' '.join(card_str(c) for c in self.board)}")
        elif self.stage == "flop":
            self.stage = "turn"
            self.board.append(self.deck.pop())
            self.log.append(f"Turn: {card_str(self.board[3])}")
        elif self.stage == "turn":
            self.stage = "river"
            self.board.append(self.deck.pop())
            self.log.append(f"River: {card_str(self.board[4])}")
        elif self.stage == "river":
            self._showdown()
            return

        seats = sorted(self.players.keys())
        idx = seats.index(self.dealer_seat)
        for off in range(1, len(seats) + 1):
            s = seats[(idx + off) % len(seats)]
            if not self.players[s].folded and not self.players[s].all_in:
                self.current_seat = s
                return
        while self.stage in ("flop", "turn"):
            if self.stage == "flop":
                self.stage = "turn"
                self.board.append(self.deck.pop())
            elif self.stage == "turn":
                self.stage = "river"
                self.board.append(self.deck.pop())
        self._showdown()

    def _advance_seat(self):
        seats = sorted(self.players.keys())
        idx = seats.index(self.current_seat) if self.current_seat in seats else 0
        for off in range(1, len(seats) + 1):
            s = seats[(idx + off) % len(seats)]
            p = self.players[s]
            if not p.folded and not p.all_in:
                self.current_seat = s
                return
        self.current_seat = None

    def _end_hand_no_showdown(self, winner: Player):
        winner.stack += self.pot
        self.log.append(f"{winner.name} wins {self.pot} chips (others folded).")
        self.last_winner_info = f"{winner.name} wins {self.pot}."
        self.stage = "done"
        self.current_seat = None

    def _showdown(self):
        living = [p for p in self.players.values() if not p.folded]
        if not living:
            self.stage = "done"
            return
        scores = {p.seat: eval_7card(p.hole_cards + self.board) for p in living}
        best = max(scores.values())
        winners = [p for p in living if scores[p.seat] == best]
        share = self.pot // len(winners)
        for w in winners:
            w.stack += share
        names = ", ".join(w.name for w in winners)
        msg = f"Showdown: {names} win{'' if len(winners)>1 else 's'} {share} each."
        self.log.append(msg)
        self.last_winner_info = msg
        self.stage = "done"
        self.current_seat = None

    def snapshot(self, requester_seat: Optional[int] = None,
                 reveal_all: bool = False) -> dict:
        with self.lock:
            ps = []
            for s in sorted(self.players.keys()):
                p = self.players[s]
                show_cards = reveal_all or (requester_seat == s) or \
                             (self.stage == "done" and not p.folded)
                ps.append({
                    "seat": p.seat,
                    "name": p.name,
                    "stack": p.stack,
                    "cur_bet": p.cur_bet,
                    "total_bet": p.total_bet,
                    "folded": p.folded,
                    "all_in": p.all_in,
                    "has_acted": p.has_acted,
                    "hole_cards": ([card_str(c) for c in p.hole_cards]
                                   if show_cards else None),
                })
            snap = {
                "session_id": self.session_id,
                "players": ps,
                "board": [card_str(c) for c in self.board],
                "pot": self.pot,
                "high_bet": self.high_bet,
                "stage": self.stage,
                "current_seat": self.current_seat,
                "dealer_seat": self.dealer_seat,
                "sb_chips": self.sb_chips,
                "bb_chips": self.bb_chips,
                "starting_stack": self.starting_stack,
                "last_winner_info": self.last_winner_info,
                "log": self.log[-30:],
                "coach": self.coach_advice if reveal_all else None,
                "num_seated": len(self.players),
            }
            return snap


class LucyCoach:
    def __init__(self, lucy_binary: str, model: str,
                 hand_abstraction: str = "v3ir",
                 abstraction: str = "fcpa"):
        self.proc: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
        if lucy_binary and model and os.path.exists(lucy_binary) and os.path.exists(model):
            self.proc = subprocess.Popen(
                [lucy_binary, "--serve", model,
                 "--abstraction", abstraction,
                 "--hand-abstraction", hand_abstraction],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        self.lucy_binary = lucy_binary
        self.model = model
        self.hand_abstraction = hand_abstraction

    def advise(self, table: PokerTable, seat: int) -> dict:
        if not self.proc:
            return {"available": False, "reason": "Lucy --serve unavailable."}

        with table.lock:
            p = table.players.get(seat)
            if not p or p.folded or p.all_in:
                return {"available": False, "reason": "Player not active."}
            if table.current_seat != seat:
                return {"available": False, "reason": "Not this player's turn."}

            seats = sorted(table.players.keys())
            num_active = sum(1 for sp in table.players.values()
                             if not sp.folded)
            is_hu = num_active == 2

            stage_map = {"preflop": 1, "flop": 2, "turn": 3, "river": 4}
            stage = stage_map.get(table.stage, 1)

            dealer = 0
            client_pos = 1

            req = {
                "num_players": 2,
                "player_id": client_pos,
                "dealer": dealer,
                "sb": table.sb_chips,
                "bb": table.bb_chips,
                "starting_stack": table.starting_stack,
                "hole_cards": p.hole_cards,
                "board_cards": list(table.board),
                "num_board": len(table.board),
                "stage": stage,
                "history": [],
            }

        try:
            with self.lock:
                self.proc.stdin.write(json.dumps(req) + "\n")
                self.proc.stdin.flush()
                resp_line = self.proc.stdout.readline()
            resp = json.loads(resp_line)
            if not resp.get("ok", True):
                return {"available": False,
                        "reason": resp.get("error", "Lucy error")}

            probs = resp.get("probabilities", [])
            legal = resp.get("legal_actions", [])
            return {
                "available": True,
                "is_heads_up": is_hu,
                "multi_way_caveat": (None if is_hu else
                    "Lucy is HU-trained; tighten ranges ~15% in multi-way pots."),
                "probabilities": probs,
                "legal_actions": legal,
                "raw_response": resp,
            }
        except Exception as e:
            return {"available": False, "reason": f"Lucy query failed: {e}"}


app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)))
TABLES: dict[str, PokerTable] = {}
TABLES_LOCK = threading.Lock()
COACH: Optional[LucyCoach] = None


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/create_table", methods=["POST"])
def create_table():
    data = request.json or {}
    sb = int(data.get("sb", 1))
    bb = int(data.get("bb", 2))
    stack = int(data.get("starting_stack", 200))
    sid = secrets.token_urlsafe(12)
    with TABLES_LOCK:
        TABLES[sid] = PokerTable(sid, sb_chips=sb, bb_chips=bb,
                                  starting_stack=stack)
    return jsonify({"session_id": sid})


@app.route("/api/table/<sid>/join", methods=["POST"])
def join_table(sid):
    data = request.json or {}
    name = data.get("name", "Anon")[:24]
    with TABLES_LOCK:
        t = TABLES.get(sid)
    if not t:
        return jsonify({"error": "no such table"}), 404
    try:
        seat = t.join(name)
        return jsonify({"seat": seat})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/table/<sid>/start_hand", methods=["POST"])
def start_hand(sid):
    with TABLES_LOCK:
        t = TABLES.get(sid)
    if not t:
        return jsonify({"error": "no such table"}), 404
    t.start_hand()
    return jsonify({"ok": True})


@app.route("/api/table/<sid>/state")
def get_state(sid):
    with TABLES_LOCK:
        t = TABLES.get(sid)
    if not t:
        return jsonify({"error": "no such table"}), 404
    seat_param = request.args.get("seat")
    reveal = request.args.get("reveal", "0") == "1"
    seat = int(seat_param) if seat_param is not None else None
    snap = t.snapshot(requester_seat=seat, reveal_all=reveal)
    if reveal and COACH and t.current_seat is not None:
        snap["coach_for_seat"] = t.current_seat
        snap["coach_advice"] = COACH.advise(t, t.current_seat)
    return jsonify(snap)


@app.route("/api/table/<sid>/act", methods=["POST"])
def act(sid):
    data = request.json or {}
    seat = int(data.get("seat", -1))
    action = data.get("action", "")
    amount = int(data.get("amount", 0))
    with TABLES_LOCK:
        t = TABLES.get(sid)
    if not t:
        return jsonify({"error": "no such table"}), 404
    try:
        t.act(seat, action, amount)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/table/<sid>/leave", methods=["POST"])
def leave_table(sid):
    data = request.json or {}
    seat = int(data.get("seat", -1))
    with TABLES_LOCK:
        t = TABLES.get(sid)
    if not t:
        return jsonify({"error": "no such table"}), 404
    t.leave(seat)
    return jsonify({"ok": True})


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--lucy-binary", default="")
    p.add_argument("--lucy-model", default="")
    p.add_argument("--hand-abstraction", default="v3ir")
    p.add_argument("--port", type=int, default=5050)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    global COACH
    if args.lucy_binary and args.lucy_model:
        COACH = LucyCoach(args.lucy_binary, args.lucy_model,
                          hand_abstraction=args.hand_abstraction)
        print(f"[demo] coach attached: {args.lucy_model}", file=sys.stderr)
    else:
        print("[demo] table-only (no coach)", file=sys.stderr)

    print(f"[demo] http://{args.host}:{args.port}", file=sys.stderr)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
