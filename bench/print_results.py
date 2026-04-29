import glob, json, os
for f in sorted(glob.glob("results/*/*/*/seed=1.json")):
    d = json.load(open(f))
    matchup = f.split(os.sep)[1]
    li = int(f.split(os.sep)[2].split("=")[1])
    bi = int(f.split(os.sep)[3].split("=")[1])
    se = d["mbb_per_hand_a_se"] or 0
    print(f"{matchup:32s} lucy={li:>6d} baseline={bi:>6d}  "
          f"A_mbb/h={d['mbb_per_hand_a']:+9.1f}  "
          f"95%CI=[{d['ci95_low']:+8.1f}, {d['ci95_high']:+8.1f}]  "
          f"n={d['num_hands']:>5d}  "
          f"{d['seconds']:.1f}s")
