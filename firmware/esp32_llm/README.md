# ESP32-S3 on-chip inference

This sketch runs the 28.9M-parameter PLE TinyLM on an ESP32-S3 N16R8. The model
lives in the custom `model` flash partition at `0x110000`; the tied
embedding/output head is staged in PSRAM at boot.

## Build and flash

```bash
bash scripts/deploy.sh
```

Use `scripts/deploy.sh` for verified current builds. It runs both host gates,
compiles at global `-O3`, flashes the model and then the firmware, and prints
the fingerprint the board should report back. Compilation happens before either
flash, so a build failure cannot leave new weights under old firmware.

The steps below are what the script does, for running a piece of it by hand.

### Host gates

```bash
cc -O3 -Wall -Wextra -o /tmp/verify firmware/host_verify/verify.c -lm
/tmp/verify firmware/model/model.bin firmware/model/golden.txt

cc -O3 -Wall -Wextra -DLLM_INT8_ACT=1 -o /tmp/staging firmware/host_verify/staging_verify.c -lm
/tmp/staging firmware/model/model.bin
```

The first checks the exact int4 path against the PyTorch golden; the second the
staged int8 kernel and platform hooks, which is what the device runs.

To regenerate the model itself:

```bash
uv run python src/export.py
```

Build the device firmware with Arduino ESP32 core 3.3.10:

```bash
arduino-cli compile \
  --fqbn 'esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi,DebugLevel=info' \
  --build-property compiler.optimization_flags=-O3 \
  --build-path /tmp/esp32-llm-build \
  firmware/esp32_llm
```

## Flash and run

Replace the port if the board enumerates under a different device name:

```bash
arduino-cli upload \
  -p /dev/cu.usbmodem2101 \
  --fqbn 'esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi,DebugLevel=info' \
  --input-dir /tmp/esp32-llm-build \
  firmware/esp32_llm

esptool.py --chip esp32s3 --port /dev/cu.usbmodem2101 --baud 921600 \
  write_flash 0x110000 firmware/model/model.bin

arduino-cli monitor -p /dev/cu.usbmodem2101 --config baudrate=115200
```

The model payload only needs reflashing after a new export. Firmware-only
changes can be uploaded without rewriting the model partition.

The model used for the measurements below has SHA-256:

```text
21067f5d78113f6c64a8720b05ff7e5c774dab0276797a522f81a6797253d97c
```

Expected boot diagnostics for the current artifact:

```text
model: V=32768 D=96 L=6 H=4 F=66 P=128  (mapped 15.6 MB)
norms  -> SRAM   20 vectors
hot set-> SRAM   21128 B dynamic + 8192 B static = 29320 B managed
weights-> PSRAM  44 tensors int8, 4.22 MB allocated
build: bytes=14912332 fp=82c5b847 sram=29320B psram=4.22MB
free: sram 294 KB | psram 3.71 MB
```

`fp` is FNV-1a over the mapped image; `deploy.sh` prints the same value for the
file it flashed and the two must agree. A `FATAL:` line means an allocation
missed its intended tier; initialization stops rather than run elsewhere.

Measured on this board, 200 tokens:

| | published | current |
|---|---:|---:|
| ms/token (compute) | 102.9 | **94.9** |
| tok/s (attached serial) | ~9.5 | **9.88** |
| output head | 57.6 | 59.4 |
| attention | 25.6 | 20.5 |
| PLE path | 8.5 | 6.4 |
| FFN | 6.9 | 6.5 |
| input | 4.4 | 2.2 |

The head is staged int8 and split across both LX7 cores, and is
PSRAM-bandwidth-bound. int8 activations cost +0.0004 nats of validation CE over
32,768 predictions (2.4816 -> 2.4820, ppl 11.96 both). The fp32 host golden
matches PyTorch to 1e-5.
