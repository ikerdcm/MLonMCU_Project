import sys, time
import serial.tools.list_ports as lp
from ppk2_api.ppk2_api import PPK2_MP
import numpy as np

dur = float(sys.argv[1]) if len(sys.argv) > 1 else 32.0
out = sys.argv[2] if len(sys.argv) > 2 else '/tmp/u5_ppk2.npz'
vdd = 3.3

def open_ppk(p):
    k = PPK2_MP(p)
    try: k.stop_measuring()           # in case a prior run left it streaming
    except Exception: pass
    time.sleep(0.5)
    try: k.ser.reset_input_buffer()
    except Exception: pass
    for _ in range(4):
        try:
            m = k.get_modifiers()
            if m is not None: return k, m
        except Exception:
            try: k.ser.reset_input_buffer()
            except Exception: pass
        time.sleep(0.4)
    return k, None

# find the PPK2 control port (the interface that answers GET_META)
ppk = None
for p in [x.device for x in lp.comports() if x.vid == 0x1915]:
    k, m = open_ppk(p)
    if m is not None:
        ppk = k; print("PPK2 control port:", p); break
    print("  no modifiers on", p)
    try: k.ser.close()
    except Exception: pass
if ppk is None:
    print("ERROR: no responsive PPK2 port"); sys.exit(1)

ppk.use_ampere_meter()
ppk.current_vdd = int(vdd * 1000)          # ampere mode: board supplies V; just satisfy the guard
ppk.toggle_DUT_power("OFF")
time.sleep(0.3)
ppk.start_measuring()                       # spawns background fetcher (drains serial continuously)

samples, digital = [], []
t0 = time.time(); powered = False
while time.time() - t0 < dur:
    d = ppk.get_data()
    if d != b'':
        s, raw = ppk.get_samples(d)
        samples += s; digital += raw
    if not powered and time.time() - t0 > 0.7:
        ppk.toggle_DUT_power("ON")           # power MCU through shunt -> boot -> 6-inference sequence
        powered = True; print("DUT power ON at t=%.2fs" % (time.time() - t0))
    time.sleep(0.02)

ppk.stop_measuring()
ppk.toggle_DUT_power("OFF")

cur  = np.asarray(samples, dtype=np.float64)             # microamps
draw = np.asarray(digital, dtype=np.uint8)               # raw logic byte (8 channels packed)
n = min(len(cur), len(draw)); cur = cur[:n]; draw = draw[:n]
fs = n / dur
np.savez(out, current_ua=cur, digital_raw=draw, fs=fs, vdd=vdd, dur=dur)

print(f"samples={n}  approx_fs={fs:.0f}Hz  mean_I={cur.mean():.0f}uA  median_I={np.median(cur):.0f}uA  max_I={cur.max():.0f}uA")
print("--- per-channel digital activity (looking for 6 pulses) ---")
for ch in range(8):
    bit = (draw >> ch) & 1
    tr = int(np.abs(np.diff(bit.astype(int))).sum())
    hi = float(bit.mean())
    flag = "  <== candidate marker" if 2 <= tr <= 40 else ""
    print(f"  D{ch}: transitions={tr:6d}  high_frac={hi:.3f}{flag}")
print("saved:", out)
