#!/usr/bin/env python3
"""Dump strategy across canonical hands for each Lucy variant model."""
import json, subprocess, os, sys

ROOT = "/work/pi_machta_umass_edu/victordesouz_umass_edu/lucy-bench"
LUCY_BIN = f"{ROOT}/lucy-v3/build/bin/PokerBotMAIF"
sys.path.insert(0, f"{ROOT}/lucy-v3")

# Canonical test hands.
HANDS = [
    # (label, hole, board, stage, history, hand_abs)
    ("AA preflop btn",      ["Ah","As"], [],            "preflop", []),
    ("KK preflop btn",      ["Kh","Ks"], [],            "preflop", []),
    ("AK off preflop btn",  ["Ah","Kd"], [],            "preflop", []),
    ("72o preflop btn",     ["7h","2d"], [],            "preflop", []),
    ("AA on AKK flop, BB", ["Ah","Ad"], ["Ac","Kh","Kd"], "flop",
                                  [{"player":1,"action":1},{"player":0,"action":1}]),
    ("FlushDraw on flop, BB",  ["Ah","Kh"], ["2h","7h","Td"], "flop",
                                  [{"player":1,"action":1},{"player":0,"action":1}]),
    ("Pair-of-twos on AK6, BB", ["2h","2d"], ["Ah","Kc","6d"], "flop",
                                  [{"player":1,"action":1},{"player":0,"action":1}]),
    ("Air on dry flop, BB",     ["7h","2d"], ["Ah","Kc","6d"], "flop",
                                  [{"player":1,"action":1},{"player":0,"action":1}]),
]

def query(model, ha, hand_label, hole, board, stage, history):
    p = subprocess.Popen(
        [LUCY_BIN, "--serve", model, "--abstraction", "fcpa",
         "--hand-abstraction", ha],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1, cwd=ROOT)
    pid = 1 if stage == "preflop" else 0
    req = {"player_id": pid, "num_players": 2, "dealer": 0,
           "sb": 1, "bb": 2, "starting_stack": 200,
           "stage": stage, "hole": hole, "board": board, "history": history,
           "abstraction": "fcpa"}
    p.stdin.write(json.dumps(req) + "\n")
    p.stdin.flush()
    line = p.stdout.readline()
    p.stdin.close()
    p.wait(timeout=5)
    return json.loads(line)

VARIANTS = [
    ("v3-baseline",  "v1"),
    ("v3-v2bucket",  "v2"),
    ("v3-dcfr",      "v2"),
    ("v3-os",        "v2"),
    ("v3-dcfr-os",   "v2"),
]

print(f"{'hand':35s} | {'variant':14s} | F     C     P     A     | (action probs aligned with [F,C,P,A]; missing = illegal)")
print("-" * 130)
ITERS = 1000000
for label, hole, board, stage, history in HANDS:
    for v, ha in VARIANTS:
        path = f"{ROOT}/models-v3/{v}/iters={ITERS}/seed=1.dat"
        if not os.path.exists(path):
            print(f"{label:35s} | {v:14s} | (model missing)")
            continue
        try:
            r = query(path, ha, label, hole, board, stage, history)
            la = r["legal_actions"]; pr = r["probabilities"]
            slots = ["    -", "    -", "    -", "    -"]
            for a, p in zip(la, pr):
                slots[a] = f"{p*100:5.1f}"
            print(f"{label:35s} | {v:14s} | {slots[0]} {slots[1]} {slots[2]} {slots[3]} |  bucket={r['infoset'].split('|')[0]}")
        except Exception as e:
            print(f"{label:35s} | {v:14s} | ERROR: {e}")
    print()
