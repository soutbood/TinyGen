# ⚡ TinyGen — a Fashion-MNIST CVAE running entirely on an ESP32-S2

[![Stars](https://img.shields.io/github/stars/soutbood/TinyGen?style=flat&logo=github&logoColor=white&label=stars&color=blue)](https://github.com/soutbood/TinyGen)
[![Forks](https://img.shields.io/github/forks/soutbood/TinyGen?style=flat&logo=github&logoColor=white&label=forks&color=teal)](https://github.com/soutbood/TinyGen)
[![Watchers](https://img.shields.io/github/watchers/soutbood/TinyGen?style=flat&logo=github&logoColor=white&label=watchers&color=orange)](https://github.com/soutbood/TinyGen)
[![Issues](https://img.shields.io/github/issues/soutbood/TinyGen?style=flat&logo=github&logoColor=white&label=issues&color=red)](https://github.com/soutbood/TinyGen/issues)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Chip](https://img.shields.io/badge/ESP32--S2-240%20MHz%20%7C%202%20MB%20PSRAM-red)](#hardware-requirements)
[![C++](https://img.shields.io/badge/firmware-C%2B%2B-blue)](src/esp32.ino)
[![PyTorch](https://img.shields.io/badge/training-PyTorch%20%2F%20LibTorch-orange)](#future-plans)
[![Cloud](https://img.shields.io/badge/cloud-0%25-brightgreen)](#)

TinyGen is a firmware I wrote (with some ai help for low-level specific tasks like 
random thermal thing) for the ESP32-S2 (the ones with PSRAM) that
generates little 32×32 pixel image of clothings — T-shirts, sandals,
sneakers, bags,etc. — with a Conditional Variational Autoencoder (CVAE)
that runs completely on the chip. No cloud, no API.

You open the esp32's website , type *"shoe"* (or just click
the Sneaker button), and about 30 seconds later the ESP32 sends back a sneaker
that it generated, sampled from its own latent space. It's slow, it's
tiny, and it's entirely self-contained.
## WAIT TILL END OR JUST SKIP IT. ##
<video src="https://codeberg.org/soutbood/TinyGen/media/branch/main/docs/demo.mp4" controls width="600"></video>


---

## What it can do

- Runs the whole CVAE decoder on a single-core 240 MHz Xtensa LX7. No TFLite,
  no external runtime, no network calls.
- Ships the weights as INT8 (under 500 KB) in a SPIFFS partition, loads them
  into PSRAM at boot. Activations stay float so the pictures don't get weird.
- Every generation is genuinely new: the latent vector is seeded from the
  ESP32's hardware thermal-noise RNG to make the generated image different.
- Serves a website (all HTML/CSS/JS inline, zero external assets) with
  a grayscale and thermal-colormap views of the output.
- Reachable at `http://tinygen.local` via mDNS.

---

## How it works

### Training happens on a PC, not on the chip

I trained the model on Fashion-MNIST (padded to 32×32, pixels
quantized to 24 intensity levels) in C++ with LibTorch, on a GeForce 940MX with
2 GB of VRAM. It worked after 4 hours of trying to find the compatible versions.
Only the decoder ever goes to the ESP32; the encoder is only used during
export to compute a "prototype" latent vector per class.

### Quantize while exporting

`export_model.py` takes a checkpoint, checks that the layer names and shapes
match what the firmware expects, quantizes every decoder weight tensor to
symmetric per-tensor INT8, and spits out two files:

- `data/model_weights.bin` — the packed weights + scales (~486 KB)
- `src/model_meta.h` — layer count, dimensions, class names 

### What actually runs on the chip

| Stage | Operation |
|---|---|
| Conditioning | 16-d class embedding ⊕ 32-d latent `z` → 48-d condition vector |
| Projection | `Linear(48 → 3072)` → reshape to 192×4×4 |
| Upsample 1 | nearest 2× → `Conv3×3 (192→128)` → GroupNorm(16) → **FiLM** → ReLU |
| Upsample 2 | nearest 2× → `Conv3×3 (128→64)` → GroupNorm(8) → **FiLM** → ReLU |
| Upsample 3 | nearest 2× → `Conv3×3 (64→32)` → GroupNorm(8) → **FiLM** → ReLU |
| Refine | `Conv3×3 (32→32)` → ReLU |
| Head | `Conv1×1 (32→24)` → per-pixel argmax → 32×32 class-index image |

The condition vector (latent + class) is injected at every resolution through
FiLM — basically a per-channel scale and shift predicted from the condition.
Without it, the decoder will ignore `z` and just draw the average member
of each class. With it, the latent actually matters.

### Serving 

An `ESPAsyncWebServer` WebSocket endpoint accepts (the only server i know about)
`{"action":"generate","prompt":"..."}`, maps the prompt to one of the 10
Fashion-MNIST classes, samples `z`, runs the decoder in its own FreeRTOS task,
and streams the image back as hex-encoded JSON. The browser decodes it and
paints both canvases. with it the colorization is done on the browser so that
the esp32-s2 can focus on generation instead.

## Why INT8? The memory math

This whole decision was made because the chip: ESP32-S2 Mini had 4 MB of
flash and only2 MB of PSRAM. budget...

| Weight format | Size | Fits? |
|---|---|---|
| float32 as a C header of numbers | ~6.0 MB | ❌ Doesn't even fit in 4 MB flash. |
| float32 binary | >2.0 MB | ❌ Fills PSRAM by itself — the ~894 KB of activation buffers have nowhere to live |
| float16 binary | ~1.0 MB | ⚠️ Squeezes in (995 + <900 ≈ 1.89 MB) with ~110 KB to spare, and the S2 has no FP16 hardware anyway |
| **int8 binary (what I ship)** | **~486 KB** | ✅ 486 KB weights + 894 KB activations = 1.38 MB, leaving ~660 KB of PSRAM breathing room |

Reasons INT8 was the right call:

- PSRAM bandwidth is the real bottleneck here. The convolutions stream weights
  from PSRAM constantly, and 1 byte per weight instead of 4 cuts that traffic
  by 4×.
- It's just one 486 KB blob in SPIFFS. `pio run -t uploadfs` and done.

If you port this to (say, an esp32-s3), the export script
can emit float32 without any fuss.

---

### Memory footprint (ESP32-S2, 4 MB flash / 2 MB PSRAM)

| Component | Location | Size |
|---|---|---|
| Firmware + web UI | flash (app partition) | ~1.5 MB partition |
| INT8 weights | flash (SPIFFS) → PSRAM at boot | ~486 KB |
| Activation buffers | PSRAM (pre-allocated) | ~894 KB |
| Conv scratch tiles | internal SRAM (`DRAM_ATTR`) | ~21 KB |
| Free PSRAM during inference | — | ~660 KB |

**Speed:** ~30 s per 32×32 image at 240 MHz.

---

## Hardware requirements

- An **ESP32-S2 with ≥ 2 MB PSRAM and ≥ 4 MB flash**. I tested it on ESP32-S2 Mini (WROVER-class module).
- Boards **without PSRAM will not run this**. The activation buffers alone is triple the SRAM.
  
---

## Building & flashing

```bash
# 1. Export a trained checkpoint (see Future plans for training your own)
python export_model.py

# 2. Put your WiFi credentials at the top of src/esp32.ino (yes, really)
WIFI_SSID = your wifi ssid
WIFI_PASS = your wifi password

# 3. Flash the firmware, then the weights partition
pio run --target uploadfs
pio run --target upload

# 4. Watch it think
pio device monitor          # 115200 baud
```

Then open **`http://tinygen.local`** and click the button half a minute and done.

---

## Limitations & status

- This is a hobby project. Image quality depends entirely on the checkpoint
  you export or use mine. early epochs produce bad images (the current model is at 20 epoch tho).
- One generation at a time. One browser tab recommended i plan to add client limit to the website.
- 32×32 with 24 colors(grayscale or thermal) isn't really a product.

---

## Future plans

- Add explanation on how to train your own model i said multiple time you can use your
  own but never explain how the model's architecture is (maybe you can try extracting it
  from the main.cpp but it shouldn't work) i plan to do it eventually.
- Test it on ESP32-S3.

---

## License

Everything I wrote here is MIT (see `LICENSE`). Every third-party component
below keeps its own license — this project just links against or builds on
them.

---

## Acknowledgments

This project builds on other people's work:

- **Fashion-MNIST** — Zalando Research — <https://github.com/zalandoresearch/fashion-mnist>
- **PyTorch / LibTorch** — PyTorch team & contributors — <https://pytorch.org> even tho i wasted 4 hours on it still was helpful.
- **ESP-IDF** — Espressif Systems — <https://github.com/espressif/esp-idf>
- **Arduino core for ESP32** — Espressif & community — <https://github.com/espressif/arduino-esp32>
- **SPIFFS** — Peter Andersson — <https://github.com/pellepl/spiffs>
- **ESPAsyncWebServer & AsyncTCP** — Hristo Gochkov & ESP32Async — <https://github.com/ESP32Async/ESPAsyncWebServer>
- **ArduinoJson** — Benoît Blanchon — <https://arduinojson.org>
- **PlatformIO** — <https://platformio.org>
- Anonymous forum posters who wrote up ESP32 PSRAM/watchdog crash stories the second problem i faced.
- A certain AI chat(not agent again i am on budget) on architecture, quantization, esp core, training method and generating this ReadMe as i know nothing about it's code format and bold header link etc.
---
