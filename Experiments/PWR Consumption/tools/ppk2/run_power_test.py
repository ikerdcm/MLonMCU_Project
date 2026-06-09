#!/usr/bin/env python3
"""Automated PPK2 power test: power the DUT, capture, auto-segment inference
spikes, re-zero to the first inference, window to 30 s, compute energy/inference,
and save the trace as the colleague's `on.csv` (Timestamp(ms),Current(uA)) plus
`on.json` summary, `on_inferences.csv` per-inference, and `on.npz`.

Output goes to `Experiments/PWR Consumption/<base>.{csv,json,npz}` (base=`on` by
default) — a staging spot; move `on.csv` into the right
`<Board>/<model_vN>/<mode>/` folder afterwards.

Requires the nRF Power Profiler app CLOSED (one app owns the PPK2 at a time).

Usage: run_power_test.py <capture_dur_s> [base_name]
  capture_dur_s : wall-clock capture length. Must cover the boot offset (~6 s)
                  PLUS the 30 s window -> use >= 38 s for the 6x5s protocol.
"""
import sys, os, json, time, numpy as np
import serial.tools.list_ports as lp
from ppk2_api.ppk2_api import PPK2_MP

DUR      = float(sys.argv[1]) if len(sys.argv) > 1 else 38.0
BASE     = sys.argv[2] if len(sys.argv) > 2 else 'on'
VDD      = 3.3
FS       = 100000.0   # PPK2 native sample rate (100 kHz) -> dt = 0.01 ms exactly
WINDOW_S = 30.0       # saved record length; first inference is re-zeroed to t=0
PRE_S    = 1.0        # idle lead-in kept BEFORE the first inference (t goes -PRE_S..)

# Save into the PWR Consumption root (this script is .../PWR Consumption/tools/ppk2/).
PWR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT      = os.path.join(PWR_ROOT, BASE)


def open_ppk(cls):
    for p in [x.device for x in lp.comports() if x.vid == 0x1915]:
        k = cls(p)
        try: k.stop_measuring()
        except Exception: pass
        time.sleep(0.5)
        try: k.ser.reset_input_buffer()
        except Exception: pass
        for _ in range(4):
            try:
                if k.get_modifiers() is not None:
                    print("PPK2:", p); return k
            except Exception:
                try: k.ser.reset_input_buffer()
                except Exception: pass
            time.sleep(0.4)
        try: k.ser.close()
        except Exception: pass
    return None


def progress(elapsed, total, powered):
    frac = min(1.0, elapsed / total)
    n = 32; filled = int(n * frac)
    bar = "#" * filled + "-" * (n - filled)
    state = "INFERENCE ON " if powered else "settling...  "
    sys.stdout.write(f"\r[{bar}] {frac*100:5.1f}%  {elapsed:5.1f}/{total:.0f}s  {state}")
    sys.stdout.flush()


k = open_ppk(PPK2_MP)
if k is None:
    print("ERROR: PPK2 not reachable (is the nRF app still open?)"); sys.exit(1)
k.use_ampere_meter(); k.current_vdd = int(VDD*1000)   # Ampere meter: board self-powered, PPK2 in series
k.toggle_DUT_power("OFF"); time.sleep(0.4)
k.start_measuring()
cur = []
t0 = time.time(); powered = False; last_draw = 0.0
while True:
    el = time.time() - t0
    if el >= DUR: break
    d = k.get_data()
    if d != b'':
        s, _ = k.get_samples(d); cur += s    # drain whenever data is ready
    else:
        time.sleep(0.001)                    # only nap when the buffer is empty
    if not powered and el > 0.6:
        k.toggle_DUT_power("ON"); powered = True   # power the DUT -> board boots & runs inferences
    if el - last_draw >= 0.1:
        progress(el, DUR, powered); last_draw = el
progress(DUR, DUR, powered); print()
k.stop_measuring(); k.toggle_DUT_power("OFF")

cur = np.asarray(cur, float)
print(f"captured {len(cur)} samples (~{len(cur)/FS:.1f}s @ 100kHz)")

# --- auto-segment inference spikes on the FULL capture (sleep baseline << inference) ---
w = max(1, int(0.003*FS)); sm = np.convolve(cur, np.ones(w)/w, 'same')
base = np.percentile(sm, 20); peak = np.percentile(sm, 99)
thr = base + 0.4*(peak - base)
above = sm > thr
edges = np.diff(above.astype(int)); rise = np.where(edges == 1)[0]; fall = np.where(edges == -1)[0]
raw = []
for r in rise:
    f = fall[fall > r]
    if len(f) == 0: continue
    f = f[0]
    if not (0.040 <= (f - r)/FS <= 0.150): continue  # keep ~71ms inferences; drop the post-loop blip and any boot blob
    if cur[r:f].max() > 50000: continue              # drop the ~210mA power-on inrush (inferences peak ~22mA)
    raw.append((int(r), int(f)))

# Calibrate the TRUE PPK2 sample rate: the API assumes 100 kHz but the device is
# ~0.2% off (skews time + energy). The firmware fires inferences exactly PERIOD_S
# apart (SysTick-accurate), so spacing-in-samples / PERIOD_S = real Hz.
PERIOD_S = 5.0   # must match KWS20_CFG_POWER_PERIOD_MS / 1000
if len(raw) >= 2:
    spacing = float(np.median(np.diff([r for r, _ in raw])))
    FS = round(spacing / PERIOD_S, 1)
    print(f"calibrated PPK2 rate: {FS:.1f} Hz (from {PERIOD_S}s spike spacing; nominal 100000)")

# Keep the full un-windowed capture so a missing 6th inference can be diagnosed.
cur_full = cur

# --- re-zero on the first inference + window (PRE_S idle lead-in, then WINDOW_S):
# inferences at t = 0, 5, 10, 15, 20, 25 s; trace runs from -PRE_S s ---
off = 0   # samples between window start and the first inference (t=0)
if raw:
    i0 = raw[0][0]
    start = max(0, i0 - int(PRE_S * FS))
    off = i0 - start
    w_end = min(len(cur), start + int((PRE_S + WINDOW_S) * FS))
    if len(cur) < i0 + int(WINDOW_S * FS):
        print(f"WARNING: capture too short for a full {WINDOW_S:.0f}s window after "
              f"first inference (need >= {(i0/FS)+WINDOW_S:.1f}s capture) -> trace clipped")
    cur = cur[start:w_end]
    raw = [(r - start, f - start) for (r, f) in raw if 0 <= r - start < len(cur)]
else:
    # RUN mode: idle == active, no spikes to anchor on. Re-zero to board power-on
    # and skip ~200ms so the power-on inrush + the 0-mA "DUT off" lead are trimmed.
    print("no spikes (run mode) -- windowing from board power-on, skipping inrush")
    pw = np.where(cur > 5000)[0]
    if len(pw):
        start = min(len(cur) - 1, pw[0] + int(0.20 * FS))
        cur = cur[start:start + int(WINDOW_S * FS)]

# Trim a leading power-on inrush (~210mA) if it landed inside the window — it's a
# capacitor-charge artifact, not inference current (which peaks ~22mA). Intermittent
# because boot timing shifts it in/out of the -PRE_S pre-roll.
lead = np.where(cur[:int(2 * FS)] > 50000)[0]
if len(lead):
    cut = lead[-1] + int(0.05 * FS)
    cur = cur[cut:]
    off -= cut                      # keep the first inference at t=0

# --- save outputs ---
np.savez(OUT + ".npz", current_ua=cur, fs=FS, vdd=VDD)

# raw trace CSV (colleague's format Timestamp(ms),Current(uA); 0.01 ms grid; first inference at t=0, so the lead-in is negative)
ts_ms = (np.arange(len(cur)) - off) * (1000.0 / FS)
np.savetxt(OUT + ".csv", np.column_stack([ts_ms, cur]),
           delimiter=",", header="Timestamp(ms),Current(uA)", comments="",
           fmt=["%.2f", "%.3f"])

spikes = []
for (r, f) in raw:
    seg = cur[r:f]
    spikes.append(dict(t_s=round((r - off)/FS, 3), dur_ms=round((f-r)/FS*1e3, 1),
                       mean_uA=round(float(seg.mean()), 0),
                       E_uJ=round(float(seg.mean()*1e-6*VDD*(len(seg)/FS)*1e6), 1)))
durs = np.array([s['dur_ms'] for s in spikes]); Es = np.array([s['E_uJ'] for s in spikes])
# Average active power over the powered region (the headline number in RUN mode,
# where idle == active and there are no separable spikes).
powered = cur[cur > 5000]
avg_uA = float(powered.mean()) if powered.size else float(cur.mean())
res = dict(label=BASE, fs_hz=int(FS), n_inferences=len(spikes),
           avg_active_mA=round(avg_uA/1000, 2), avg_active_mW=round(avg_uA*1e-6*VDD*1000, 2),
           idle_mA=round(base/1000, 2), idle_mW=round(base*1e-6*VDD*1000, 2),
           infer_mA=round(peak/1000, 2), infer_mW=round(peak*1e-6*VDD*1000, 2),
           infer_ms_mean=round(float(durs.mean()), 1) if len(durs) else None,
           E_per_inf_uJ_mean=round(float(Es.mean()), 0) if len(Es) else None,
           E_per_inf_uJ_std=round(float(Es.std()), 0) if len(Es) else None,
           E_excess_uJ=round(float((Es.mean() - base*1e-6*VDD*durs.mean()*1e-3*1e6)), 0) if len(Es) else None)
json.dump(res, open(OUT + ".json", "w"), indent=2)

with open(OUT + "_inferences.csv", "w") as fcsv:
    fcsv.write("inference,t_s,dur_ms,mean_uA,E_uJ\n")
    for i, s in enumerate(spikes):
        fcsv.write(f"{i},{s['t_s']},{s['dur_ms']},{s['mean_uA']:.0f},{s['E_uJ']}\n")

print(json.dumps(res, indent=2))
print("inference times (s):", [s['t_s'] for s in spikes])
print("saved:")
for ext in (".csv", ".json", "_inferences.csv", ".npz"):
    print("  ", OUT + ext)

# If we didn't get the expected 6, dump the full un-windowed capture for diagnosis.
if len(spikes) != 6:
    np.savez(OUT + "_full.npz", current_ua=cur_full, fs=FS, vdd=VDD)
    print(f"\n!! got {len(spikes)} inferences (expected 6) -> wrote {OUT}_full.npz "
          f"({len(cur_full)/FS:.1f}s full capture) for diagnosis")

print(f"\n-> move {BASE}.csv into Experiments/PWR Consumption/<Board>/<model_vN>/<mode>/")
