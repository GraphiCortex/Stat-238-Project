from config import SESSION_DIR, BASE

patterns = [
    f"{BASE}.whl",
    f"{BASE}.par",
    f"{BASE}.eeg*",
    f"{BASE}.res.*",
    f"{BASE}.clu.*",
]

print("SESSION_DIR =", SESSION_DIR)
for pat in patterns:
    hits = sorted(SESSION_DIR.glob(pat))
    print(f"{pat:15s} -> {len(hits)} files")
    for h in hits[:5]:
        print("   ", h.name)
    if len(hits) > 5:
        print("    ...")
