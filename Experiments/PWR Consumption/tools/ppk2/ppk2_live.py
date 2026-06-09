#!/usr/bin/env python3
"""Live PPK2 power dashboard. Owns the PPK2 (source 3.3V, powers the DUT),
streams current to a browser at http://127.0.0.1:8077 via SSE. Requires the
nRF Power Profiler app CLOSED."""
import threading, time, json, collections, numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import serial.tools.list_ports as lp
from ppk2_api.ppk2_api import PPK2_MP

VDD = 3.3; BIN_S = 0.005; WIN_S = 6.0; MAXB = int(WIN_S / BIN_S)

def open_ppk():
    for p in [x.device for x in lp.comports() if x.vid == 0x1915]:
        k = PPK2_MP(p)
        try: k.stop_measuring()
        except Exception: pass
        time.sleep(0.5)
        try: k.ser.reset_input_buffer()
        except Exception: pass
        for _ in range(4):
            try:
                if k.get_modifiers() is not None: return k, p
            except Exception:
                try: k.ser.reset_input_buffer()
                except Exception: pass
            time.sleep(0.4)
        try: k.ser.close()
        except Exception: pass
    return None, None

ppk, port = open_ppk()
if ppk is None:
    print("PPK2 not reachable (is the nRF app still open?)"); raise SystemExit(1)
print("PPK2:", port)
ppk.use_ampere_meter(); ppk.current_vdd = int(VDD*1000); ppk.toggle_DUT_power("ON")   # Ampere meter (blue)
ppk.start_measuring()

binmax = collections.deque(maxlen=MAXB); binmean = collections.deque(maxlen=MAXB)
lock = threading.Lock(); G = {'charge_mC': 0.0}

def reader():
    fs = 100000.0; nper = int(BIN_S*fs); raw = []
    while True:
        d = ppk.get_data()
        if d != b'':
            s, _ = ppk.get_samples(d); raw.extend(s)
        while len(raw) >= nper:
            c = np.asarray(raw[:nper]); del raw[:nper]
            with lock:
                binmax.append(float(c.max())/1000.0); binmean.append(float(c.mean())/1000.0)
                G['charge_mC'] += float(c.mean())*1e-6*BIN_S*1000
        time.sleep(0.004)

threading.Thread(target=reader, daemon=True).start()

def snapshot():
    with lock:
        mx = list(binmax); mn = list(binmean); ch = G['charge_mC']
    st = {'mA': 0, 'idle_mA': 0, 'infer_mA': 0, 'inf_per_s': 0, 'E_inf_uJ': 0, 'charge_mC': round(ch, 1)}
    if mn:
        a = np.array(mn); mxa = np.array(mx)
        idle = float(np.percentile(a, 20)); infer = float(np.percentile(mxa, 99))
        thr = idle + 0.4*(infer-idle); above = mxa > thr
        nsp = int((np.diff(above.astype(int)) == 1).sum()); dur = len(a)*BIN_S
        st.update(mA=round(a[-1], 2), idle_mA=round(idle, 2), infer_mA=round(infer, 2),
                  inf_per_s=round(nsp/dur, 2) if dur else 0,
                  E_inf_uJ=round(infer*VDD*0.072*1000, 0))
    return mx, st

HTML = """<!doctype html><html><head><meta charset=utf-8><title>U5 live power</title>
<style>body{font-family:-apple-system,Arial;margin:0;background:#0e1116;color:#e6e6e6}
.top{display:flex;gap:14px;padding:12px 18px;flex-wrap:wrap}
.tile{background:#1b2027;border:1px solid #2a313b;border-radius:10px;padding:10px 14px;min-width:118px}
.tile .v{font-size:24px;font-weight:700}.tile .l{font-size:11px;color:#8b95a3;text-transform:uppercase;letter-spacing:.04em}
.v.blue{color:#4ea1ff}.v.amber{color:#f0a93b}.v.green{color:#3ad07a}
canvas{display:block;margin:6px 18px;background:#11161d;border:1px solid #2a313b;border-radius:10px;width:calc(100% - 36px)}
h1{font-size:15px;margin:14px 18px 0;font-weight:600}.sub{color:#8b95a3;font-size:12px;margin:2px 18px 8px}</style></head>
<body><h1>STM32U5 — live power (PPK2, source 3.3 V) <span id=dot style="color:#3ad07a">●</span></h1>
<div class=sub>each spike = one DS-CNN inference · 6 s window · auto-refresh</div>
<div class=top>
<div class=tile><div class="v blue" id=mA>–</div><div class=l>current (mA)</div></div>
<div class=tile><div class="v" id=idle>–</div><div class=l>idle / sleep (mA)</div></div>
<div class=tile><div class="v amber" id=infer>–</div><div class=l>inference peak (mA)</div></div>
<div class=tile><div class="v green" id=rate>–</div><div class=l>inferences / s</div></div>
<div class=tile><div class="v" id=einf>–</div><div class=l>~energy / inf (µJ)</div></div>
<div class=tile><div class="v" id=charge>–</div><div class=l>charge (mC)</div></div>
</div>
<canvas id=c width=1400 height=360></canvas>
<script>
const cv=document.getElementById('c'),ctx=cv.getContext('2d'),YMAX=25;
function draw(mx){const W=cv.width,H=cv.height,pad=40;ctx.clearRect(0,0,W,H);
 ctx.strokeStyle='#2a313b';ctx.fillStyle='#8b95a3';ctx.font='11px Arial';ctx.lineWidth=1;
 for(let m=0;m<=YMAX;m+=5){let y=H-pad-(H-2*pad)*m/YMAX;ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(W-8,y);ctx.stroke();ctx.fillText(m+' mA',4,y+3);}
 if(!mx.length)return;const n=mx.length;ctx.strokeStyle='#4ea1ff';ctx.lineWidth=1.4;ctx.beginPath();
 for(let i=0;i<n;i++){let x=pad+(W-pad-8)*i/(n-1);let y=H-pad-(H-2*pad)*Math.min(mx[i],YMAX)/YMAX;i?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.stroke();}
const es=new EventSource('/stream');let on=1;
es.onmessage=e=>{const d=JSON.parse(e.data);draw(d.mx);const s=d.stats;
 mA.textContent=s.mA;idle.textContent=s.idle_mA;infer.textContent=s.infer_mA;rate.textContent=s.inf_per_s;einf.textContent=s.E_inf_uJ;charge.textContent=s.charge_mC;
 dot.style.opacity=(on^=1)?1:.3;};
es.onerror=()=>{dot.style.color='#b3261e';};
</script></body></html>"""

class Hd(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == '/':
            self.send_response(200); self.send_header('Content-Type', 'text/html'); self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path == '/stream':
            self.send_response(200); self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache'); self.end_headers()
            try:
                while True:
                    mx, st = snapshot()
                    self.wfile.write(("data: " + json.dumps({'mx': [round(v, 2) for v in mx], 'stats': st}) + "\n\n").encode())
                    self.wfile.flush(); time.sleep(0.15)
            except Exception: pass

print("serving http://127.0.0.1:8077")
ThreadingHTTPServer(('127.0.0.1', 8077), Hd).serve_forever()
