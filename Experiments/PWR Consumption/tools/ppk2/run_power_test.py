#!/usr/bin/env python3
"""Automated PPK2 power test: power the DUT, capture, auto-segment inference
spikes, compute energy/inference, save npz + summary JSON. Requires the nRF
Power Profiler app CLOSED (one app owns the PPK2 at a time)."""
import sys, json, time, numpy as np
import serial.tools.list_ports as lp
from ppk2_api.ppk2_api import PPK2_MP, PPK2_API

DUR   = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
LABEL = sys.argv[2] if len(sys.argv) > 2 else 'u5_int8'
OUT   = f"/tmp/pwr_{LABEL}"
VDD   = 3.3

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

k = open_ppk(PPK2_MP)
if k is None:
    print("ERROR: PPK2 not reachable (is the nRF app still open?)"); sys.exit(1)
k.use_ampere_meter(); k.current_vdd = int(VDD*1000)   # Ampere meter (blue): board self-powered, PPK2 in series
k.toggle_DUT_power("OFF"); time.sleep(0.4)
k.start_measuring()
cur = []
t0 = time.time(); powered = False
while time.time() - t0 < DUR:
    d = k.get_data()
    if d != b'':
        s, _ = k.get_samples(d); cur += s
    if not powered and time.time() - t0 > 0.6:
        k.toggle_DUT_power("ON"); powered = True
    time.sleep(0.02)
k.stop_measuring(); k.toggle_DUT_power("OFF")

WINDOW_S = 30.0   # saved record length; first inference is re-zeroed to t=0

cur = np.asarray(cur, float); fs = len(cur)/DUR

# --- auto-segment inference spikes (sleep baseline << inference) ---
# Run detection on the FULL capture so the boot/settle offset doesn't matter;
# we re-zero to the first inference afterwards.
w = max(1, int(0.003*fs)); sm = np.convolve(cur, np.ones(w)/w, 'same')
base = np.percentile(sm, 20); peak = np.percentile(sm, 99)
thr = base + 0.4*(peak - base)
above = sm > thr
edges = np.diff(above.astype(int)); rise = np.where(edges == 1)[0]; fall = np.where(edges == -1)[0]
raw = []
for r in rise:
    f = fall[fall > r]
    if len(f) == 0: continue
    f = f[0]
    if not (0.020 <= (f - r)/fs <= 0.30): continue   # keep ~inference-width; drop <20ms blips and the >300ms boot/settle blob
    raw.append((r, f))

# --- re-zero on the first inference + window to WINDOW_S (official protocol:
# inferences at t = 0, 5, 10, 15, 20, 25 s in a 30 s record) ---
if raw:
    i0 = raw[0][0]                                   # sample idx of the first inference
    w_n = int(WINDOW_S * fs)
    w_end = min(len(cur), i0 + w_n)
    cur = cur[i0:w_end]                              # windowed, re-zeroed trace
    raw = [(r - i0, f - i0) for (r, f) in raw if r - i0 < len(cur)]
else:
    print("WARNING: no inference spikes detected — saving full trace, no re-zero")

np.savez(OUT + ".npz", current_ua=cur, fs=fs, vdd=VDD)

# --- raw trace CSV (colleague's "Timestamp(ms),Current(uA)" format) ---
# Plots + analyze_power_peaks.py consume this directly. t=0 is the first inference.
ts_ms = np.arange(len(cur)) * (1000.0 / fs)
np.savetxt(OUT + "_trace.csv", np.column_stack([ts_ms, cur]),
           delimiter=",", header="Timestamp(ms),Current(uA)", comments="",
           fmt=["%.4f", "%.3f"])

spikes = []
for (r, f) in raw:
    seg = cur[r:f]
    spikes.append(dict(t_s=round(r/fs, 3), dur_ms=round((f-r)/fs*1e3, 1),
                       mean_uA=round(float(seg.mean()), 0),
                       E_uJ=round(float(seg.mean()*1e-6*VDD*(len(seg)/fs)*1e6), 1)))
durs = np.array([s['dur_ms'] for s in spikes]); Es = np.array([s['E_uJ'] for s in spikes])
res = dict(label=LABEL, fs_hz=round(fs), n_inferences=len(spikes),
           idle_mA=round(base/1000, 2), idle_mW=round(base*1e-6*VDD*1000, 2),
           infer_mA=round(peak/1000, 2), infer_mW=round(peak*1e-6*VDD*1000, 2),
           infer_ms_mean=round(float(durs.mean()), 1) if len(durs) else None,
           E_per_inf_uJ_mean=round(float(Es.mean()), 0) if len(Es) else None,
           E_per_inf_uJ_std=round(float(Es.std()), 0) if len(Es) else None,
           E_excess_uJ=round(float((Es.mean() - base*1e-6*VDD*durs.mean()*1e-3*1e6)), 0) if len(Es) else None)
json.dump(res, open(OUT + ".json", "w"), indent=2)

# --- per-inference summary CSV (one row per detected spike) ---
with open(OUT + "_inferences.csv", "w") as fcsv:
    fcsv.write("inference,t_s,dur_ms,mean_uA,E_uJ\n")
    for i, s in enumerate(spikes):
        fcsv.write(f"{i},{s['t_s']},{s['dur_ms']},{s['mean_uA']:.0f},{s['E_uJ']}\n")

print(json.dumps(res, indent=2))
print("first few spikes:", spikes[:4])
print("saved:", OUT + ".npz", OUT + ".json", OUT + "_trace.csv", OUT + "_inferences.csv")
