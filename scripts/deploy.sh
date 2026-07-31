#!/usr/bin/env bash
# Flash the TinyStories model + firmware to an ESP32-S3.
#
# The board holds ONE model at a time: the 14.9MB image claims the whole `model`
# partition. There is no repartitioning trick.
# pipefail: every step is piped through tail/grep/tr, and `set -e` alone does
# not see a failure on the left of a pipe.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# TinyStories artifact paths. artifacts/ holds downloaded or exported model
# files and is ignored; generated/ holds headers built from them.
SKETCH=firmware/esp32_tinystories
ARTIFACTS=${ARTIFACTS:-artifacts/tinystories}
MODEL=${MODEL:-$ARTIFACTS/model.bin}
TOKENIZER=${TOKENIZER:-$ARTIFACTS/tokenizer.json}
GOLDEN=${GOLDEN:-$ARTIFACTS/golden.txt}
VOCAB_H=$SKETCH/generated/vocab.h
PART_OFFSET=0x110000

FQBN='esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi,DebugLevel=info'

# -O3, overriding the Arduino core default of -Os. The runtime carries no
# per-function optimization attributes; this flag is the whole configuration.
OPT_FLAGS='-O3'

# --- locate tools without hardcoding a home directory -----------------------
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1" >&2; exit 1; }; }
need arduino-cli

find_esptool() {
  for c in esptool esptool.py; do
    command -v "$c" >/dev/null 2>&1 && { echo "$c"; return; }
  done
  # Arduino bundles it; the version directory changes between core releases, so
  # glob rather than pin. ARDUINO_DATA_DIR overrides for non-default installs.
  local data="${ARDUINO_DATA_DIR:-$HOME/Library/Arduino15}"
  [ -d "$data" ] || data="${ARDUINO_DATA_DIR:-$HOME/.arduino15}"
  local found
  found=$(ls -d "$data"/packages/esp32/tools/esptool_py/*/esptool 2>/dev/null | sort -V | tail -1 || true)
  [ -n "$found" ] && { echo "$found"; return; }
  echo "esptool not found. Install it (pip install esptool) or set ARDUINO_DATA_DIR." >&2
  exit 1
}
ESPTOOL=$(find_esptool)

# `|| true`: a failing glob would otherwise abort the assignment under `set -e`,
# before the message below can print.
PORT=${PORT:-$(ls /dev/cu.usbmodem* 2>/dev/null | head -1 || true)}
[ -n "$PORT" ] || { echo "no /dev/cu.usbmodem* found; plug the board in, or set PORT=..." >&2; exit 1; }

test -f "$MODEL" || {
  echo "$MODEL missing." >&2
  echo "Fetch the released model into $ARTIFACTS/, or export one with:" >&2
  echo "  uv run python src/export.py <checkpoint-tag>" >&2
  exit 1; }
test -f "$TOKENIZER" || {
  echo "$TOKENIZER missing. The decode table is built from it, so it must ship" >&2
  echo "with the model." >&2
  exit 1; }

# Rebuild the decode header from the tokenizer being deployed, every time. The
# firmware only checks that VOCAB_N equals the model's output_vocab, so a stale
# header from a different tokenizer with the same entry count would pass that
# check and decode every token wrongly.
echo "=== generate $VOCAB_H from $TOKENIZER ==="
uv run python "$SKETCH/tools/generate_vocab.py" \
  --tokenizer "$TOKENIZER" --out "$VOCAB_H" 2>&1 | grep wrote

CFLAGS='-O3 -Wall -Wextra'

# The golden is produced by src/export.py and is not part of a released model,
# so a clone that only fetched model.bin will not have one. Skip visibly rather
# than fail: staging_verify below needs no golden and still runs.
if [ -f "$GOLDEN" ]; then
  echo "=== host verify: exact int4 path vs PyTorch golden ==="
  cc $CFLAGS -o /tmp/llm_verify runtime/host_verify/verify.c -lm
  /tmp/llm_verify "$MODEL" "$GOLDEN" 2>&1 | tail -2
else
  echo "=== host verify: SKIPPED, no golden at $GOLDEN ==="
fi

# The device runs the staged int8 kernel through the platform hooks. verify.c
# does not reach that code - it exercises the exact int4 path - so without
# this the thing actually executing on the board has no host gate.
echo "=== host verify: int8 staging + platform hooks ==="
cc $CFLAGS -DLLM_INT8_ACT=1 -o /tmp/llm_staging runtime/host_verify/staging_verify.c -lm
/tmp/llm_staging "$MODEL" 2>&1 | tail -3

# Compile and verify before writing either image: a build failure after the
# model flash would leave new weights under old firmware.
BUILD_DIR="${TMPDIR:-/tmp}/esp32ai-build-$(basename "$SKETCH")"
echo "=== compile $SKETCH ($OPT_FLAGS) ==="
arduino-cli compile --fqbn "$FQBN" \
  --build-property "compiler.optimization_flags=$OPT_FLAGS" \
  --build-path "$BUILD_DIR" \
  "$SKETCH" 2>&1 | tail -2

echo "=== flash model ($(du -h "$MODEL" | cut -f1)) -> $PORT @ $PART_OFFSET ==="
"$ESPTOOL" --chip esp32s3 --port "$PORT" --baud 921600 \
  write_flash "$PART_OFFSET" "$MODEL" 2>&1 | tr '\r' '\n' | tail -2

# --input-dir: `upload` does not rebuild, so point it at the build just made.
echo "=== upload firmware ==="
arduino-cli upload -p "$PORT" --fqbn "$FQBN" --input-dir "$BUILD_DIR" "$SKETCH" 2>&1 | tail -1

# Compute the same FNV-1a fingerprint printed by the firmware.
FP=$(python3 -c '
import sys
d = open(sys.argv[1], "rb").read()
h = 2166136261
for b in d:
    h ^= b; h = (h * 16777619) & 0xFFFFFFFF
print("%08x" % h)' "$MODEL")

echo
echo "flashed : $MODEL"
echo "expect  : fp=$FP  bytes=$(wc -c < "$MODEL" | tr -d ' ')"
echo "the board prints 'build: bytes=... fp=...' at boot - both must match."
