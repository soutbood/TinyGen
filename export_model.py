# export_model.py
# squishes a trained checkpoint into something the esp32-s2 can swallow:
#   data/model_weights.bin  - int8 weights + per-layer scales (flashed to SPIFFS)
#   src/model_meta.h        - tiny header so the firmware knows counts & names
#
# usage: put checkpoint_20.pt next to this file and run
#        python export_model.py

import torch
import numpy as np
import struct
import gzip
import os

print("=" * 60)
print("  TinyGen weight exporter (int8 edition)")
print("=" * 60)

# ---- which checkpoint ----
# hardcoded to epoch 20 since that's the one i'm shipping right now.
# want a different epoch? just rename your file to match this line
# (my trainer saves them as checkpoint_epoch_XX.pt, i renamed mine)
ckpt = "checkpoint_20.pt"
if not os.path.exists(ckpt):
    print(f"can't find {ckpt} in this folder. copy it in first")
    exit(1)
print(f"\nusing {ckpt}")

# ---- load ----
# jit.load yells a deprecation warning at me every single time, but it's the
# only thing that reliably reads what the C++ trainer saves. so we live with it
print("\n[1] loading...")
try:
    state_dict = torch.jit.load(ckpt, map_location='cpu').state_dict()
except Exception:
    state_dict = torch.load(ckpt, map_location='cpu')
print(f"    {len(state_dict)} tensors in there")

# decoder tensors in the exact order the firmware reads them.
# rename a layer in the trainer = rename it here too. learned that the hard way
weight_keys = [
    "decoder.class_embed.weight",
    "decoder.emb_proj.weight",
    "decoder.emb_proj.bias",
    "decoder.projection.weight",
    "decoder.projection.bias",
    "decoder.up1_conv.weight",
    "decoder.up1_conv.bias",
    "decoder.up1_gn.weight",
    "decoder.up1_gn.bias",
    "decoder.film1.weight",
    "decoder.film1.bias",
    "decoder.up2_conv.weight",
    "decoder.up2_conv.bias",
    "decoder.up2_gn.weight",
    "decoder.up2_gn.bias",
    "decoder.film2.weight",
    "decoder.film2.bias",
    "decoder.up3_conv.weight",
    "decoder.up3_conv.bias",
    "decoder.up3_gn.weight",
    "decoder.up3_gn.bias",
    "decoder.film3.weight",
    "decoder.film3.bias",
    "decoder.refine.weight",
    "decoder.refine.bias",
    "decoder.classifier.weight",
    "decoder.classifier.bias",
]

missing = [k for k in weight_keys if k not in state_dict]
if missing:
    print("    checkpoint doesn't match what the firmware expects, missing:")
    for m in missing:
        print("      -", m)
    exit(1)

# ---- class latents ----
# the chip has no encoder, so we bake one average latent per class into the
# weights file. generations on-device start from these + noise
print("\n[2] class latents...")

class EncoderLite(torch.nn.Module):
    """minimal copy of the trainer's encoder, just enough to get mu out"""
    def __init__(self):
        super().__init__()
        self.class_embed = torch.nn.Embedding(10, 16)
        self.emb_proj = torch.nn.Linear(16, 16)
        self.conv1 = torch.nn.Conv2d(1, 64, 4, stride=2, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, 64)
        self.conv2 = torch.nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.gn2 = torch.nn.GroupNorm(16, 128)
        self.conv3 = torch.nn.Conv2d(128, 192, 4, stride=2, padding=1)
        self.gn3 = torch.nn.GroupNorm(16, 192)
        self.fc_mean = torch.nn.Linear(192 * 4 * 4 + 16, 32)

    def forward(self, x, cls):
        import torch.nn.functional as F
        cv = torch.relu(self.emb_proj(self.class_embed(cls)))
        x = torch.relu(self.gn1(self.conv1(x)))
        x = torch.relu(self.gn2(self.conv2(x)))
        x = torch.relu(self.gn3(self.conv3(x)))
        x = torch.cat([x.flatten(1), cv], dim=1)
        return self.fc_mean(x)

encoder = EncoderLite()
enc_state = {k.replace('encoder.', ''): v for k, v in state_dict.items()
             if 'encoder' in k and 'emb_drop' not in k}
try:
    encoder.load_state_dict(enc_state, strict=True)
    encoder.eval()
    print("    encoder loaded fine")
except Exception as e:
    print(f"    couldn't load encoder ({e}), falling back to random latents")
    encoder = None

# fallback if anything goes sideways: random latents still generate,
# they just don't look like much
class_latents = (np.random.randn(10, 32) * 0.3).astype(np.float32)

if encoder is not None and os.path.exists('train-images-idx3-ubyte.gz'):
    # average the encoder's mu over ~200 real images per class
    with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
        struct.unpack('>IIII', f.read(16))
        raw = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
    with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as f:
        struct.unpack('>II', f.read(8))
        lbls = np.frombuffer(f.read(), dtype=np.uint8)

    padded = np.zeros((raw.shape[0], 32, 32), dtype=np.float32)
    padded[:, 2:30, 2:30] = raw.astype(np.float32) / 255.0
    imgs_t = torch.from_numpy(padded).unsqueeze(1)
    lbls_t = torch.from_numpy(lbls.astype(np.int64))

    with torch.no_grad():
        for c in range(10):
            pick = imgs_t[lbls_t == c][:200]
            cls_t = torch.full((pick.size(0),), c, dtype=torch.long)
            class_latents[c] = encoder(pick, cls_t).mean(dim=0).numpy()
    print("    computed from real training images")
else:
    print("    no training gz here, using random latents (fine for testing)")

# ---- quantize + pack ----
print("\n[3] quantizing to int8...")
os.makedirs("data", exist_ok=True)

scales, quantized = [], []

def to_int8(arr):
    """symmetric per-tensor quantize: scale = max|w| / 127"""
    flat = arr.flatten().astype(np.float32)
    m = np.max(np.abs(flat))
    if m < 1e-8:      # all-zero tensor (has happened), don't divide by zero
        m = 1e-8
    s = m / 127.0
    q = np.clip(np.round(flat / s), -127, 127).astype(np.int8)
    return s, q

for key in weight_keys:
    s, q = to_int8(state_dict[key].numpy())
    scales.append(s)
    quantized.append(q)
    print(f"    {key.replace('decoder.', ''):24s} {str(list(state_dict[key].shape)):20s} scale={s:.6f}")

# latents get the same treatment, they live at the very end of the file
s, q = to_int8(class_latents)
scales.append(s)
quantized.append(q)
print(f"    {'class_latents':24s} [10, 32]             scale={s:.6f}")

# file layout (the firmware reads it exactly like this, don't reshuffle):
#   uint32  num_layers
#   float32 scale[num_layers]
#   int8    zero_point[num_layers]   (always 0 with symmetric quant, kept for clarity)
#   uint32  size[num_layers]
#   int8    data... (all tensors concatenated)
n_layers = len(quantized)
with open("data/model_weights.bin", "wb") as f:
    f.write(struct.pack('<I', n_layers))
    for s in scales:
        f.write(struct.pack('<f', s))
    for _ in range(n_layers):
        f.write(struct.pack('<b', 0))
    for q in quantized:
        f.write(struct.pack('<I', q.size))
    for q in quantized:
        f.write(q.tobytes())

bin_size = os.path.getsize("data/model_weights.bin")
raw_size = sum(q.size for q in quantized) * 4
print(f"\n    wrote data/model_weights.bin: {bin_size/1024:.1f} KB "
      f"(float32 would've been {raw_size/1024:.1f} KB)")

# ---- meta header ----
print("\n[4] writing src/model_meta.h...")
os.makedirs("src", exist_ok=True)

names = [k.replace('decoder.', '').replace('.', '_') for k in weight_keys] + ["class_latents"]
with open("src/model_meta.h", "w") as f:
    f.write("// generated by export_model.py - do not edit by hand\n")
    f.write("#pragma once\n\n")
    f.write(f"#define NUM_WEIGHT_LAYERS {n_layers}\n")
    f.write("#define MODEL_LATENT_DIM 32\n")
    f.write("#define MODEL_EMB_DIM 16\n")
    f.write("#define MODEL_COND_DIM 48\n")
    f.write("#define MODEL_NUM_CLASSES 10\n")
    f.write("#define MODEL_NUM_COLORS 24\n")
    f.write("#define MODEL_IMG_SIZE 32\n")
    f.write("#define MODEL_C1 64\n")
    f.write("#define MODEL_C2 128\n")
    f.write("#define MODEL_C3 192\n\n")
    f.write("// order of tensors inside model_weights.bin:\n")
    for i, nm in enumerate(names):
        f.write(f"// [{i:2d}] {nm}\n")
    f.write("\nstatic const char* CLASS_NAMES[] = {\n")
    for nm in ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]:
        f.write(f'    "{nm}",\n')
    f.write("};\n")

print("\ndone. now:")
print("    pio run --target upload      (firmware)")
print("    pio run --target uploadfs    (this bin -> SPIFFS)")