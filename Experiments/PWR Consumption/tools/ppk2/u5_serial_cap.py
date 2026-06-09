import serial, time, sys
port = sys.argv[1] if len(sys.argv) > 1 else '/dev/tty.usbmodem11303'
baud = 115200
dur  = float(sys.argv[2]) if len(sys.argv) > 2 else 35.0
t0 = time.time()
try:
    s = serial.Serial(port, baud, timeout=1)
except Exception as e:
    print("OPEN FAIL", port, e); sys.exit(1)
with s, open('/tmp/u5_bench.log', 'w') as f:
    while time.time() - t0 < dur:
        line = s.readline().decode('ascii', 'replace').strip()
        if line:
            out = f"{time.time()-t0:6.2f}s  {line}"
            print(out, flush=True); f.write(out + '\n'); f.flush()
print("CAPTURE DONE", flush=True)
