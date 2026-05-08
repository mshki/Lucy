from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import requests
except ImportError:
    print("missing 'requests'; pip install requests", file=sys.stderr)
    sys.exit(1)


SLUMBOT_BASE = "https://www.slumbot.com"

SB_CHIPS = 1
BB_CHIPS = 2
STARTING_STACK_CHIPS = 200

SLUMBOT_RANKS = "23456789TJQKA"
SLUMBOT_SUITS = "cdhs"


def slumbot_card_to_int(card: str) -> int:
    if len(card) != 2:
        raise ValueError(card)
    return SLUMBOT_RANKS.index(card[0]) * 4 + SLUMBOT_SUITS.index(card[1])


def parse_board(board_str: str) -> list[int]:
    if not board_str.strip():
        return []
    cleaned = board_str.replace("/", " ").split()
    return [slumbot_card_to_int(c) for c in cleaned if len(c) == 2]


@dataclass
class HandState:
    hole_cards: list[int]
    board: list[int]
    stage: int
    client_pos: int
    pot: int
    cur_bet: list[int]
    high_bet: int
    stack: list[int]
    history: list[dict]
    is_terminal: bool
    winner: Optional[int] = None


def _parse_action_stream(action: str, hole_cards: list[int],
                         board: list[int], client_pos: int,
                         starting_stack: int = STARTING_STACK_CHIPS,
                         sb: int = SB_CHIPS, bb: int = BB_CHIPS) -> HandState:
    streets = action.split("/")

    pot = sb + bb
    cur_bet = [sb, bb]
    total_bet = [sb, bb]
    stack = [starting_stack - sb, starting_stack - bb]
    high_bet = bb

    stage = 1
    history: list[dict] = []
    folded = [False, False]

    def reset_street():
        nonlocal cur_bet, high_bet
        cur_bet = [0, 0]
        high_bet = 0

    def first_to_act(stage_):
        return 0 if stage_ == 1 else 1

    actor = first_to_act(stage)
    is_terminal = False
    winner: Optional[int] = None

    for street_idx, street in enumerate(streets):
        if street_idx > 0:
            stage = street_idx + 1
            reset_street()
            actor = first_to_act(stage)

        i = 0
        while i < len(street):
            ch = street[i]
            if ch == "f":
                folded[actor] = True
                history.append({"player": actor, "action": "f", "amount": 0})
                is_terminal = True
                winner = 1 - actor
                break
            elif ch == "c":
                call_amt = high_bet - cur_bet[actor]
                if call_amt > 0:
                    chips = min(call_amt, stack[actor])
                    cur_bet[actor]   += chips
                    total_bet[actor] += chips
                    stack[actor]     -= chips
                    pot              += chips
                history.append({"player": actor, "action": "c", "amount": call_amt})
                i += 1
                actor = 1 - actor
            elif ch == "b":
                j = i + 1
                while j < len(street) and street[j].isdigit():
                    j += 1
                amount = int(street[i+1:j]) if j > i + 1 else 0
                add = max(0, amount - cur_bet[actor])
                add = min(add, stack[actor])
                cur_bet[actor]   += add
                total_bet[actor] += add
                stack[actor]     -= add
                pot              += add
                if cur_bet[actor] > high_bet:
                    high_bet = cur_bet[actor]
                history.append({"player": actor, "action": "b", "amount": amount})
                i = j
                actor = 1 - actor
            else:
                i += 1

        if is_terminal:
            break

    return HandState(
        hole_cards=hole_cards, board=board, stage=stage,
        client_pos=client_pos, pot=pot, cur_bet=cur_bet, high_bet=high_bet,
        stack=stack, history=history, is_terminal=is_terminal, winner=winner,
    )


class LucyServer:
    def __init__(self, lucy_binary: str, model: str,
                 hand_abstraction: str = "v3ir",
                 abstraction: str = "fcpa"):
        self.proc = subprocess.Popen(
            [lucy_binary, "--serve", model,
             "--abstraction", abstraction,
             "--hand-abstraction", hand_abstraction],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )

    def query(self, state: HandState, num_players: int = 2,
              dealer: int = 0, sb: int = SB_CHIPS, bb: int = BB_CHIPS,
              starting_stack: int = STARTING_STACK_CHIPS) -> dict:
        req = {
            "num_players": num_players,
            "player_id": state.client_pos,
            "dealer": dealer,
            "sb": sb, "bb": bb, "starting_stack": starting_stack,
            "hole_cards": state.hole_cards,
            "board_cards": state.board,
            "num_board": len(state.board),
            "stage": state.stage,
            "history": [{"player": h["player"],
                         "action": _action_to_lucy_id(h["action"], h["amount"])}
                        for h in state.history],
        }
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self):
        if self.proc:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()


def _action_to_lucy_id(action: str, amount: int) -> int:
    if action == "f":
        return 0
    if action == "c":
        return 1
    if action == "b":
        return 3 if amount >= 100 else 2
    return 1


class SlumbotClient:
    def __init__(self, base_url: str = SLUMBOT_BASE,
                 username: str = "", password: str = ""):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        self.token: Optional[str] = None
        if username and password:
            self.login(username, password)

    def login(self, username: str, password: str):
        r = self.session.post(self.base + "/api/login",
                              json={"username": username, "password": password},
                              timeout=20)
        r.raise_for_status()
        data = r.json()
        self.token = data.get("token") or data.get("session_token")

    def new_hand(self) -> dict:
        body = {"token": self.token} if self.token else {}
        r = self.session.post(self.base + "/api/new_hand", json=body, timeout=20)
        r.raise_for_status()
        return r.json()

    def act(self, incr: str) -> dict:
        body = {"incr": incr}
        if self.token:
            body["token"] = self.token
        r = self.session.post(self.base + "/api/act", json=body, timeout=20)
        r.raise_for_status()
        return r.json()


def lucy_action_to_acpc(lucy_action_id: int, state: HandState,
                       sb: int = SB_CHIPS, bb: int = BB_CHIPS,
                       starting_stack: int = STARTING_STACK_CHIPS) -> str:
    me = state.client_pos
    if lucy_action_id == 0:
        return "f"
    if lucy_action_id == 1:
        return "c"
    if lucy_action_id == 2:
        call_amt = state.high_bet - state.cur_bet[me]
        if call_amt == 0:
            amount = state.cur_bet[me] + max(state.pot, bb)
        else:
            amount = state.high_bet + state.pot + call_amt
        amount = min(amount, state.cur_bet[me] + state.stack[me])
        return f"b{amount}"
    if lucy_action_id == 3:
        return f"b{state.cur_bet[me] + state.stack[me]}"
    return "c"


def play_session(num_hands: int, lucy_binary: str, lucy_model: str,
                 lucy_abstraction: str = "fcpa",
                 lucy_hand_abstraction: str = "v3ir",
                 username: str = "", password: str = "",
                 verbose: bool = True,
                 base_url: str = SLUMBOT_BASE) -> dict:
    client = SlumbotClient(base_url=base_url, username=username, password=password)
    lucy = LucyServer(lucy_binary, lucy_model,
                      hand_abstraction=lucy_hand_abstraction,
                      abstraction=lucy_abstraction)

    total_winnings = 0
    hands_played = 0
    errors = 0

    try:
        for hand_idx in range(num_hands):
            try:
                resp = client.new_hand()
            except Exception as e:
                print(f"[slumbot] new_hand: {e}", file=sys.stderr)
                errors += 1
                continue

            client_pos = int(resp.get("client_pos", 0))
            hole = [slumbot_card_to_int(c) for c in resp.get("hole_cards", [])]
            action_stream = resp.get("action", "")
            board_str = resp.get("board", "")

            while True:
                board = parse_board(board_str)
                state = _parse_action_stream(action_stream, hole, board, client_pos)
                if state.is_terminal:
                    break

                if state.history:
                    next_actor = 1 - state.history[-1]["player"]
                else:
                    next_actor = 0 if state.stage == 1 else 1

                if next_actor != client_pos:
                    break

                lucy_resp = lucy.query(state)
                probs = lucy_resp.get("probabilities", [])
                legal = lucy_resp.get("legal_actions", [])
                if not probs or not legal:
                    incr = "c"
                else:
                    r = random.random()
                    acc = 0.0
                    chosen = legal[0]
                    for la, p in zip(legal, probs):
                        acc += p
                        if r <= acc:
                            chosen = la
                            break
                    incr = lucy_action_to_acpc(int(chosen), state)

                try:
                    resp = client.act(incr)
                except Exception as e:
                    print(f"[slumbot] act: {e}", file=sys.stderr)
                    errors += 1
                    break
                action_stream = resp.get("action", action_stream)
                board_str = resp.get("board", board_str)

            chip_delta = int(resp.get("winnings", 0))
            total_winnings += chip_delta
            hands_played += 1
            if verbose and hand_idx % 50 == 0:
                mbb = (total_winnings / max(1, hands_played)) * 1000 / BB_CHIPS
                print(f"[slumbot] {hand_idx+1}/{num_hands} cum={total_winnings:+d} avg={mbb:+.1f} mbb/hand",
                      file=sys.stderr)
    finally:
        lucy.close()

    mbb = (total_winnings / max(1, hands_played)) * 1000 / BB_CHIPS
    return {
        "num_hands": hands_played,
        "errors": errors,
        "total_winnings_chips": total_winnings,
        "mbb_per_hand": mbb,
        "lucy_model": lucy_model,
        "lucy_hand_abstraction": lucy_hand_abstraction,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lucy-binary", required=True)
    p.add_argument("--lucy-model", required=True)
    p.add_argument("--lucy-hand-abstraction", default="v3ir",
                   choices=["v1", "v1ir", "v2", "v3", "v3ir"])
    p.add_argument("--lucy-abstraction", default="fcpa",
                   choices=["fcpa", "legacy", "street-rich"])
    p.add_argument("--num-hands", type=int, default=1000)
    p.add_argument("--username", default="")
    p.add_argument("--password", default="")
    p.add_argument("--out")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--base-url", default=SLUMBOT_BASE)
    args = p.parse_args()

    print(f"[slumbot] {args.num_hands} hands; model={args.lucy_model}", file=sys.stderr)
    t0 = time.time()
    result = play_session(
        num_hands=args.num_hands,
        lucy_binary=args.lucy_binary,
        lucy_model=args.lucy_model,
        lucy_abstraction=args.lucy_abstraction,
        lucy_hand_abstraction=args.lucy_hand_abstraction,
        username=args.username,
        password=args.password,
        verbose=not args.quiet,
        base_url=args.base_url,
    )
    result["elapsed_seconds"] = time.time() - t0

    print(f"hands={result['num_hands']} errors={result['errors']} "
          f"chips={result['total_winnings_chips']:+d} "
          f"mbb/hand={result['mbb_per_hand']:+.1f} "
          f"elapsed={result['elapsed_seconds']:.1f}s")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
