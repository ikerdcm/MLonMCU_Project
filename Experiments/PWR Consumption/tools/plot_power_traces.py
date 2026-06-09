#!/usr/bin/env python3
"""Interactive overlay-plotter for PPK2 power traces.

A curses file browser walks `Experiments/PWR Consumption/` so you go in/out of
the per-board / per-model / per-mode folders instead of staring at one flat list.

  UP / DOWN     move cursor
  RIGHT / ENTER on a folder -> open it;  on ".." -> go up
  LEFT          go up a folder
  SPACE         toggle-select the highlighted CSV (selection persists across folders)
  ENTER         on the "[ DONE ]" row (or just press 'p') -> plot the selected traces
  q / ESC       quit

Then it plots the selected traces overlaid with proper scaling (decimated
envelope so the inference spikes survive).

Run:  python3 plot_power_traces.py        (venv with matplotlib)
Opts: --power     y-axis in mW (uses --vdd, default 3.3 V)
      --vdd V     supply voltage for the mW conversion
      --align     re-zero each trace to its own first inference spike
      --out PATH  save the figure (default: tools/power_overlay.png)
"""
import os, sys, csv, argparse
import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("matplotlib not installed. In the venv:  /tmp/ppk2venv/bin/pip install matplotlib")
    sys.exit(1)

PWR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HEADER_HINT = "Timestamp(ms)"


def is_power_csv(path):
    try:
        with open(path) as f:
            return HEADER_HINT in f.readline()
    except Exception:
        return False


# ----------------------------- curses browser -----------------------------
def browse(start_dir):
    """Walk the tree; return a list of selected abs CSV paths (or [] if cancelled)."""
    import curses

    def run(scr):
        curses.curs_set(0)
        cur_dir = start_dir
        selected = set()
        idx = 0
        top = 0

        def listing(d):
            rows = []  # (label, kind, payload)
            if os.path.abspath(d) != PWR_ROOT:
                rows.append(("../", "up", os.path.dirname(d)))
            entries = sorted(os.listdir(d))
            for e in entries:
                p = os.path.join(d, e)
                if os.path.isdir(p):
                    rows.append((e + "/", "dir", p))
            for e in entries:
                p = os.path.join(d, e)
                if os.path.isfile(p) and e.endswith(".csv") and is_power_csv(p):
                    rows.append((e, "file", p))
            rows.append(("[ DONE -> plot selected ]", "done", None))
            return rows

        rows = listing(cur_dir)
        while True:
            scr.erase()
            h, w = scr.getmaxyx()
            rel = os.path.relpath(cur_dir, PWR_ROOT)
            scr.addnstr(0, 0, f"PWR Consumption/{'' if rel=='.' else rel}", w - 1, curses.A_BOLD)
            scr.addnstr(1, 0, f"selected: {len(selected)}   "
                              "[↑↓ move  →/Enter open  ← up  Space select  "
                              "Enter@DONE/p plot  q quit]", w - 1)
            view_h = h - 3
            if idx < top: top = idx
            if idx >= top + view_h: top = idx - view_h + 1
            for i in range(top, min(len(rows), top + view_h)):
                label, kind, payload = rows[i]
                mark = " "
                if kind == "file":
                    mark = "x" if payload in selected else " "
                    line = f"[{mark}] {label}"
                elif kind == "done":
                    line = label
                else:
                    line = f"    {label}"
                attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
                if kind == "dir": attr |= curses.A_BOLD
                scr.addnstr(2 + i - top, 0, line, w - 1, attr)
            scr.refresh()

            c = scr.getch()
            if c in (ord('q'), 27):
                return []
            elif c in (curses.KEY_UP, ord('k')):
                idx = (idx - 1) % len(rows)
            elif c in (curses.KEY_DOWN, ord('j')):
                idx = (idx + 1) % len(rows)
            elif c == ord(' '):
                label, kind, payload = rows[idx]
                if kind == "file":
                    selected.discard(payload) if payload in selected else selected.add(payload)
                    idx = (idx + 1) % len(rows)
            elif c in (curses.KEY_RIGHT, curses.KEY_ENTER, 10, 13, ord('l')):
                label, kind, payload = rows[idx]
                if kind in ("dir", "up"):
                    cur_dir = payload; rows = listing(cur_dir); idx = 0; top = 0
                elif kind == "done" and c in (curses.KEY_ENTER, 10, 13):
                    return sorted(selected)
                elif kind == "done":
                    pass
            elif c in (curses.KEY_LEFT, ord('h')):
                if os.path.abspath(cur_dir) != PWR_ROOT:
                    cur_dir = os.path.dirname(cur_dir); rows = listing(cur_dir); idx = 0; top = 0
            elif c == ord('p'):
                return sorted(selected)

    try:
        return curses.wrapper(run)
    except Exception as e:
        print(f"(curses browser unavailable: {e}) — falling back to flat menu")
        return flat_menu()


def flat_menu():
    traces = []
    for dp, _, files in os.walk(PWR_ROOT):
        for fn in sorted(files):
            p = os.path.join(dp, fn)
            if fn.endswith(".csv") and is_power_csv(p):
                traces.append(p)
    traces.sort()
    for i, p in enumerate(traces, 1):
        print(f"  [{i:2d}] {os.path.relpath(p, PWR_ROOT)}")
    sel = input("\nSelect (e.g. 1,3,5 or 1-3): ").replace(" ", ",")
    out = set()
    for tok in sel.split(","):
        if not tok: continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.update(range(int(a) - 1, int(b)))
        else:
            out.add(int(tok) - 1)
    return [traces[i] for i in sorted(out) if 0 <= i < len(traces)]


# ----------------------------- plotting -----------------------------
def load_trace(path):
    t, c = [], []
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) < 2: continue
            try: t.append(float(row[0])); c.append(float(row[1]))
            except ValueError: continue
    return np.asarray(t), np.asarray(c)


def first_spike_ms(t_ms, cur):
    if len(cur) < 10: return 0.0
    base = np.percentile(cur, 20); peak = np.percentile(cur, 99.5)
    thr = base + 0.4 * (peak - base)
    above = np.where(cur > thr)[0]
    return float(t_ms[above[0]]) if len(above) else 0.0


def envelope(t_ms, cur, n_bins=4000):
    n = len(cur)
    if n <= n_bins: return t_ms, cur
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    tb = np.empty(n_bins); cb = np.empty(n_bins)
    for i in range(n_bins):
        a, b = edges[i], max(edges[i] + 1, edges[i + 1])
        j = a + int(np.argmax(cur[a:b]))
        tb[i] = t_ms[j]; cb[i] = cur[j]
    return tb, cb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", action="store_true", help="plot mW instead of mA")
    ap.add_argument("--vdd", type=float, default=3.3, help="supply V for mW conversion")
    ap.add_argument("--align", action="store_true", help="re-zero each trace to its first spike")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "power_overlay.png"))
    ap.add_argument("--full", action="store_true", help="plot raw points without decimation")
    args = ap.parse_args()

    picks = browse(PWR_ROOT)
    if not picks:
        print("nothing selected"); return

    fig, ax = plt.subplots(figsize=(13, 5))
    for path in picks:
        rel = os.path.relpath(path, PWR_ROOT)
        t_ms, cur = load_trace(path)
        if len(cur) == 0:
            print(f"  (skip empty) {rel}"); continue
        if args.align:
            t_ms = t_ms - first_spike_ms(t_ms, cur)
        y = cur * 1e-3 * args.vdd if args.power else cur * 1e-3
        if args.full:
            tb, yb = t_ms, y
        else:
            tb, yb = envelope(t_ms, y)
        ax.plot(tb / 1000.0, yb, lw=0.9, label=rel)
        print(f"  plotted {rel}: {len(cur)} pts, mean {cur.mean()*1e-3:.2f} mA, peak {cur.max()*1e-3:.2f} mA")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (mW)" if args.power else "Current (mA)")
    ax.set_title("PPK2 power traces (overlaid)" + ("  [aligned to first inference]" if args.align else ""))
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.01)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"\nsaved figure: {args.out}")
    try: plt.show()
    except Exception: pass


if __name__ == "__main__":
    main()
