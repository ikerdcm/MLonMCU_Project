import sys, numpy as np
d = np.load(sys.argv[1] if len(sys.argv) > 1 else '/tmp/u5_ppk2_v1_int8.npz')
cur = d['current_ua']; fs = float(d['fs']); vdd = float(d['vdd'])
t = np.arange(len(cur)) / fs
ms = lambda s: int(s * fs)

# smooth (5 ms) for event detection
w = ms(0.005); sm = np.convolve(cur, np.ones(w)/w, mode='same')

# boot edge = first sample MCU draws real current (>8 mA)
boot = int(np.argmax(sm > 8000))
print(f"trace: {len(cur)} samp @ {fs:.0f}Hz = {t[-1]:.1f}s | boot@{t[boot]:.2f}s")
print(f"overall: mean={cur.mean():.0f}uA  median={np.median(cur):.0f}uA  p95={np.percentile(cur,95):.0f}uA  max={cur.max():.0f}uA")

# expected 6 inferences: boot + 2s settle + k*5s ; each ~64 ms
t0 = t[boot] + 2.0
print("\n--- per-inference windows (expected at boot+2s + k*5s) ---")
energies = []
for k in range(6):
    c = t0 + k * 5.0                      # expected start
    win = cur[ms(c): ms(c + 0.064)]       # 64 ms inference window
    idle = cur[ms(c - 0.5): ms(c - 0.1)]  # idle just before
    if len(win) == 0 or len(idle) == 0:
        print(f"  inf{k}: (out of range)"); continue
    e_uj = win.mean() * 1e-6 * vdd * (len(win)/fs) * 1e6   # uJ over the window
    energies.append(e_uj)
    print(f"  inf{k} @~{c:5.2f}s : win_mean={win.mean():.0f}uA  idle={idle.mean():.0f}uA  "
          f"delta={win.mean()-idle.mean():+.0f}uA  E={e_uj:.0f}uJ")

idle_all = np.median(cur[ms(t[boot]+0.3):])     # steady-state idle/active level
print(f"\nsteady current (median post-boot) = {idle_all:.0f}uA -> power = {idle_all*1e-6*vdd*1000:.1f} mW")
print(f"energy/inference @ 64ms latency = {idle_all*1e-6*vdd*0.064*1e6:.0f} uJ (P x latency)")
if energies:
    print(f"marker-window energies: mean={np.mean(energies):.0f}uJ std={np.std(energies):.0f}uJ")
# distinct high-current periods (inference bumps?) above p90
thr = np.percentile(sm, 90)
above = sm > max(thr, idle_all*1.15)
edges = np.diff(above.astype(int)); nblk = int((edges == 1).sum())
print(f"\ncurrent bumps >15% above idle: {nblk} blocks (6 would confirm visible inferences)")
