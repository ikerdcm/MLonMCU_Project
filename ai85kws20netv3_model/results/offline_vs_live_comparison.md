# Offline vs. Live — ai85kws20netv3 auf STM32U5 und MAX78000

## STM32U5 @ 160 MHz (X-CUBE-AI, CPU)

| Variante | Modus | Energie/Inferenz | Latenz (CNN, avg) | Flash (text+data) | SRAM statisch (data+bss) |
|---|---|---|---|---|---|
| **pruned + quantized** | offline | **11,47 mJ** | 152,90 ms | 301,7 KiB | 107,7 KiB |
| | live | **11,56 mJ** | 152,92 ms | 246,9 KiB | 176,8 KiB |
| **unpruned + quantized** | offline | **12,52 mJ** | 170,76 ms | 325,4 KiB | 111,9 KiB |
| | live | **12,64 mJ** | 173,84 ms | 270,3 KiB | 180,8 KiB |
| **unpruned + unquantized** | offline | **31,34 mJ** | 397,20 ms | 793,9 KiB | 169,2 KiB |
| | live | **31,41 mJ** | 397,24 ms | 738,6 KiB | 238,1 KiB |

## MAX78000 (FTHR_RevA, CNN-Accelerator)

| Variante | Modus | Energie/Inferenz | Latenz (CNN, avg) | Flash (text+data) | SRAM statisch (data+bss) |
|---|---|---|---|---|---|
| **kws20 baseline** | offline | **251,1 µJ** | 1,850 ms | 413,1 KiB | 37,3 KiB |
| | live | **252,4 µJ** | 1,849 ms | 383,7 KiB | 48,7 KiB |
| **kws20 w90 (pruned)** | offline | **232,3 µJ** | 1,711 ms | 389,3 KiB | 37,3 KiB |
| | live | **233,9 µJ** | 1,711 ms | 360,0 KiB | 46,8 KiB |
