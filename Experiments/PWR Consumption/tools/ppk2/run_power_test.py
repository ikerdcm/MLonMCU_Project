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

cur = np.asarray(cur, float); fs = len(cur)/DUR
np.savez(OUT + ".npz", current_ua=cur, fs=fs, vdd=VDD)

# --- auto-segment inference spikes (sleep baseline << inference) ---
w = max(1, int(0.003*fs)); sm = np.convolve(cur, np.ones(w)/w, 'same')
base = np.percentile(sm, 20); peak = np.percentile(sm, 99)
thr = base + 0.4*(peak - base)
above = sm > thr
edges = np.diff(above.astype(int)); rise = np.where(edges == 1)[0]; fall = np.where(edges == -1)[0]
spikes = []
for r in rise:
    f = fall[fall > r]
    if len(f) == 0: continue
    f = f[0]
    if not (0.020 <= (f - r)/fs <= 0.30): continue   # keep ~inference-width; drop <20ms blips and the >300ms boot/settle blob
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
print(json.dumps(res, indent=2))
print("first few spikes:", spikes[:4])
print("saved:", OUT + ".npz", OUT + ".json")
