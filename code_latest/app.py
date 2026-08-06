import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageFile
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
import gradio as gr
import pywt
from scipy import stats
from scipy.ndimage import uniform_filter
from scipy.fftpack import dct as scipy_dct

ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.set_num_threads(min(8, os.cpu_count() or 4))

MODEL_DIR        = os.path.dirname(os.path.abspath(__file__))
BACKBONE_DIR     = os.path.join(MODEL_DIR, "backbone")
HYBRID_HEAD_PATH = os.path.join(MODEL_DIR, "hybrid_head.pt")
DEVICE           = "cpu"
INPUT_RESOLUTION = 512
NUM_FEATURES     = 7

FEATURE_NAMES = [
    "lbp_entropy",
    "dct_blocking",
    "gradient_cooccurrence",
    "wavelet",
    "fft_slope",
    "bayer_noise",
    "edge_sharpness",
]
FEATURE_INVERT = {
    "lbp_entropy":           False,
    "dct_blocking":          False,
    "gradient_cooccurrence": False,
    "wavelet":               False,
    "fft_slope":             True,
    "bayer_noise":           True,
    "edge_sharpness":        True,
}

print(f"Loading model on CPU...")


class ArtifactAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.attention(x)


class FeatureMLP(nn.Module):
    def __init__(self, in_d=7, hidden=64, out_d=128, drop=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(drop * 0.8),
            nn.Linear(hidden * 2, out_d),
        )
    def forward(self, x): return self.net(x)


class HybridForensicHead(nn.Module):
    def __init__(self, in_features, num_classes=2):
        super().__init__()
        self.fc1       = nn.Linear(in_features, 1024)
        self.bn1       = nn.BatchNorm1d(1024)
        self.drop1     = nn.Dropout(0.4)
        self.attention = ArtifactAttention(1024)
        self.fc2       = nn.Linear(1024, 512)
        self.bn2       = nn.BatchNorm1d(512)
        self.drop2     = nn.Dropout(0.3)
        self.fc3       = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.drop1(F.gelu(self.bn1(self.fc1(x))))
        x = self.attention(x)
        x = self.drop2(F.gelu(self.bn2(self.fc2(x))))
        return self.fc3(x)


class HybridAIDetector(nn.Module):
    def __init__(self, backbone, backbone_dim, num_features=7, feat_embed_dim=128, num_classes=2):
        super().__init__()
        self.backbone  = backbone
        self.feat_mlp  = FeatureMLP(in_d=num_features, out_d=feat_embed_dim)
        self.head      = HybridForensicHead(backbone_dim + feat_embed_dim, num_classes)

    def forward(self, pixel_values, trad_features):
        out = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            img_emb = out.pooler_output
        else:
            img_emb = out.hidden_states[-1].mean(dim=1)
        feat_emb = self.feat_mlp(trad_features)
        return self.head(torch.cat([img_emb, feat_emb], dim=-1))


def resize_siglip_embeddings(model, new_resolution):
    vision_model = None
    if hasattr(model, "vision_model"):   vision_model = model.vision_model
    elif hasattr(model, "siglip"):       vision_model = model.siglip.vision_model
    elif hasattr(model, "vit"):          vision_model = model.vit
    if vision_model is None: return

    emb        = vision_model.embeddings
    patch_size = emb.patch_size
    new_n      = (new_resolution // patch_size) ** 2
    old_n      = emb.num_patches
    if old_n == new_n: return

    old_pos = emb.position_embedding.weight.data
    dim     = old_pos.shape[-1]
    old_g   = int(old_n ** 0.5)
    new_g   = int(new_n ** 0.5)

    old_pos = old_pos.reshape(1, old_g, old_g, dim).permute(0, 3, 1, 2)
    new_pos = F.interpolate(old_pos, size=(new_g, new_g), mode="bicubic", align_corners=False)
    new_pos = new_pos.permute(0, 2, 3, 1).reshape(new_n, dim)

    new_emb = nn.Embedding(new_n, dim)
    new_emb.weight.data = new_pos.to(next(model.parameters()).device)
    emb.position_embedding = new_emb
    emb.num_patches        = new_n
    emb.image_size         = new_resolution
    emb.register_buffer("position_ids", torch.arange(new_n).expand((1, -1)))
    model.config.vision_config.image_size = new_resolution
    print(f"  Position embeddings resized → {new_resolution}x{new_resolution}")


processor = AutoImageProcessor.from_pretrained(BACKBONE_DIR, local_files_only=True)

base = AutoModelForImageClassification.from_pretrained(
    BACKBONE_DIR, local_files_only=True, ignore_mismatched_sizes=True,
    output_hidden_states=True,
)
resize_siglip_embeddings(base, INPUT_RESOLUTION)

if isinstance(base.classifier, nn.Linear):
    backbone_dim = base.classifier.in_features
elif isinstance(base.classifier, nn.Sequential):
    for layer in base.classifier:
        if isinstance(layer, nn.Linear): backbone_dim = layer.in_features; break
else:
    backbone_dim = base.config.hidden_size
base.classifier = nn.Identity()

model = HybridAIDetector(
    backbone=base, backbone_dim=backbone_dim,
    num_features=NUM_FEATURES, feat_embed_dim=128, num_classes=2,
)

ckpt = torch.load(HYBRID_HEAD_PATH, map_location="cpu", weights_only=False)
model.feat_mlp.load_state_dict(ckpt["feat_mlp"])
model.head.load_state_dict(ckpt["head"])

SCALER_MEAN = np.array(ckpt["scaler_mean"], dtype=np.float32)
SCALER_STD  = np.array(ckpt["scaler_std"],  dtype=np.float32)

del ckpt
gc.collect()

model.to(DEVICE)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
print(f"Model ready  ({backbone_dim + 128}-d fused input)")


class AdaptiveResize:
    def __init__(self, size=512):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        if abs(w - self.size) < 50 and abs(h - self.size) < 50:
            return img.resize((self.size, self.size), Image.BICUBIC)
        max_dim = max(w, h)
        if max_dim > self.size * 2:
            scale = (self.size * 2) / max_dim
            img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img.resize((self.size, self.size), Image.BICUBIC)


SIGLIP_MEAN = processor.image_mean
SIGLIP_STD  = processor.image_std

base_transforms = transforms.Compose([
    AdaptiveResize(INPUT_RESOLUTION),
    transforms.ToTensor(),
    transforms.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD),
])


def get_image_tensor(img: Image.Image) -> torch.Tensor:
    return base_transforms(img).unsqueeze(0).to(DEVICE)


def _gray(rgb):
    return (0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]).astype(np.float32)

def _gr(ch):
    return np.gradient(ch.astype(np.float64))

def _sc(a, b):
    return 1.0 if (a.std() < 1e-9 or b.std() < 1e-9) else float(np.corrcoef(a.flatten(), b.flatten())[0, 1])

def _nr(ch):
    p = Image.fromarray(np.clip(ch, 0, 255).astype(np.uint8))
    return ch.astype(np.float32) - np.array(p.filter(ImageFilter.MedianFilter(3)), dtype=np.float32)

def _sig(x, c, sc):
    return float(1 / (1 + np.exp(-sc * (float(x) - c))))

def _clip(x):
    return float(np.clip(x, 0, 1))


def feat_lbp(gray):
    g   = gray.astype(np.float32)
    pat = np.zeros_like(g, dtype=np.uint8)
    for bit, (dy, dx) in enumerate([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]):
        pat += ((g >= np.roll(np.roll(g, dy, 0), dx, 1)).astype(np.uint8) << bit)
    h = np.bincount(pat.flatten(), minlength=256).astype(float)
    h /= h.sum() + 1e-9
    ent = -np.sum(h[h > 0] * np.log2(h[h > 0] + 1e-9))
    return _clip(_sig(abs(ent - 7.5) / 0.5, 0.5, 8))


def feat_dct(gray):
    h, w = gray.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    if h8 == 0 or w8 == 0: return 0.5
    tiles = gray[:h8, :w8].reshape(h8//8, 8, w8//8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    rng   = np.random.default_rng(42)
    idx   = rng.choice(len(tiles), min(len(tiles), 400), replace=False)
    ac    = []
    for i in idx:
        d = scipy_dct(scipy_dct(tiles[i].astype(np.float64).T, norm="ortho").T, norm="ortho")
        ac.extend(d.flatten()[1:].tolist())
    ac = np.abs(np.array(ac)); nz = ac[ac >= 1]
    if len(nz) < 50: return 0.5
    fd = []
    for v in nz[:3000]:
        while v >= 10: v /= 10
        fd.append(int(v))
    fd   = np.array(fd, dtype=int)
    obs  = np.array([np.sum(fd == d) for d in range(1, 10)], dtype=float); obs /= obs.sum() + 1e-9
    benf = np.array([np.log10(1 + 1/d) for d in range(1, 10)])
    return _clip(_sig(float(np.sum((obs - benf)**2 / (benf + 1e-9))), 0.06, 35))


def feat_gc(gray):
    gy, gx = _gr(gray); gm = np.sqrt(gx**2 + gy**2)
    if gm.max() < 1e-9: return 0.5
    gq = (gm / gm.max() * 31).astype(int)
    H, W = gq.shape; coo = np.zeros((32, 32), dtype=np.float32)
    cols = min(W - 1, 300)
    a_all = gq[:, :cols].flatten()
    b_all = gq[:, 1:cols+1].flatten()
    np.add.at(coo, (a_all, b_all), 1)
    coo /= coo.sum() + 1e-9; flat = coo[coo > 0].flatten()
    return _clip(_sig(3.0 + float(np.sum(flat * np.log(flat))), 0.0, 3))


def feat_wav(gray):
    scores = []; data = gray.astype(np.float64)
    for _ in range(3):
        coeffs = pywt.dwt2(data, "haar"); _, (LH, HL, HH) = coeffs
        hh = float(np.mean(HH**2)); lh = float(np.mean(LH**2)); hl = float(np.mean(HL**2))
        scores.append(_sig(0.15 - hh / (hh + lh + hl + 1e-9), 0.0, 30)); data = coeffs[0]
    return _clip(float(np.mean(scores)))


def feat_fft(gray):
    h, w = gray.shape; mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    cy, cx = h // 2, w // 2; yi, xi = np.indices((h, w))
    r = np.sqrt((xi - cx)**2 + (yi - cy)**2).astype(int); rm_ = min(cx, cy)
    r_flat = r.flatten(); mag_flat = mag.flatten()
    rm = np.array([mag_flat[r_flat == i].mean() if np.any(r_flat == i) else np.nan for i in range(1, rm_)])
    valid = ~np.isnan(rm) & (rm > 0)
    if valid.sum() < 20: return 0.5
    slope, *_ = stats.linregress(np.log(np.arange(1, rm_)[valid]), np.log(rm[valid]))
    return _clip(_sig(abs(-slope - 2.5) / 2.5, 0.35, 8))


def feat_bn(rgb):
    nR, nG, nB = _nr(rgb[:,:,0]), _nr(rgb[:,:,1]), _nr(rgb[:,:,2])
    if (nR.std() + nG.std() + nB.std()) / 3 < 0.4: return 0.88
    def l2(n):
        if n.shape[1] < 4: return 0
        a, b = n[:,:-2].flatten(), n[:,2:].flatten()
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else 0
    ac = (abs(l2(nR)) + abs(l2(nB))) / 2
    h2, w2 = nR.shape; p = (slice(h2//4, 3*h2//4), slice(w2//4, 3*w2//4))
    rR, rG, rB = nR[p].flatten(), nG[p].flatten(), nB[p].flatten()
    cs = (0.72 if rR.std() < 1e-9 or rG.std() < 1e-9 or rB.std() < 1e-9
          else _sig(abs((_sc(rR, rG) + _sc(rB, rG)) / 2 - 0.40), 0.18, 18))
    return _clip(0.30 * _sig(0.12 - ac, 0.0, 25) + 0.70 * cs)


def feat_es(rgb):
    g = _gray(rgb); gy, gx = _gr(g); gm = np.sqrt(gx**2 + gy**2).astype(np.float32)
    H, W = gm.shape
    if H < 30 or W < 30: return 0.5
    rh, rw = H // 3, W // 3
    sm = np.array([[gm[i*rh:(i+1)*rh, j*rw:(j+1)*rw].mean() for j in range(3)] for i in range(3)])
    c  = sm[1, 1]
    pe = (np.array([sm[0,0], sm[0,2], sm[2,0], sm[2,2]]).mean() +
          np.array([sm[0,1], sm[1,0], sm[1,2], sm[2,1]]).mean()) / 2 + 1e-6
    return _clip(0.55 * _sig(1.12 - c / pe, 0, 18) + 0.45 * _sig(0.10 - sm.std() / (sm.mean() + 1e-6), 0, 45))


ALGO_MAP = {
    "lbp_entropy":           lambda rgb, g: feat_lbp(g),
    "dct_blocking":          lambda rgb, g: feat_dct(g),
    "gradient_cooccurrence": lambda rgb, g: feat_gc(g),
    "wavelet":               lambda rgb, g: feat_wav(g),
    "fft_slope":             lambda rgb, g: feat_fft(g),
    "bayer_noise":           lambda rgb, g: feat_bn(rgb),
    "edge_sharpness":        lambda rgb, g: feat_es(rgb),
}


def extract_features(pil_img: Image.Image) -> np.ndarray:
    rgb  = np.array(pil_img.resize((512, 512), Image.BICUBIC), dtype=np.float32)
    gray = _gray(rgb)
    vec  = []
    for feat in FEATURE_NAMES:
        try:
            v = float(np.clip(ALGO_MAP[feat](rgb, gray), 0, 1))
        except Exception:
            v = 0.5
        if FEATURE_INVERT[feat]:
            v = 1.0 - v
        vec.append(v)
    vec = np.array(vec, dtype=np.float32)
    return ((vec - SCALER_MEAN) / (SCALER_STD + 1e-9)).astype(np.float32)


def predict(image: Image.Image) -> dict:
    if image.mode != "RGB":
        image = image.convert("RGB")

    feat_scaled  = extract_features(image)
    feat_tensor  = torch.tensor(feat_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    pixel_tensor = get_image_tensor(image)

    with torch.no_grad():
        logits   = model(pixel_values=pixel_tensor, trad_features=feat_tensor)
        probs_np = F.softmax(logits, dim=-1).cpu().numpy()[0]

    del pixel_tensor, feat_tensor, logits
    gc.collect()

    return {
        "ai_prob":   float(probs_np[0]),
        "real_prob": float(probs_np[1]),
    }


def format_output(image_path: str) -> str:
    if not image_path:
        return _empty_state()

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return f'<div class="result-error">⚠️ Error loading image: {e}</div>'

    try:
        result = predict(image)
    except Exception as e:
        return f'<div class="result-error">⚠️ Prediction error: {e}</div>'

    ai_score   = result["ai_prob"]
    real_score = result["real_prob"]
    total = ai_score + real_score
    if total > 0:
        ai_score   /= total
        real_score /= total

    is_ai      = ai_score > real_score
    verdict    = "AI-Generated" if is_ai else "Human-Created"
    confidence = ai_score if is_ai else real_score
    icon       = "🤖" if is_ai else "📷"

    if   confidence > 0.90: conf_text = "Very High Confidence"
    elif confidence > 0.75: conf_text = "High Confidence"
    elif confidence > 0.60: conf_text = "Moderate Confidence"
    else:                   conf_text = "Low Confidence"

    ai_w   = int(ai_score   * 100)
    real_w = int(real_score * 100)

    verdict_color = "#ff4d6d" if is_ai else "#2ec4b6"
    verdict_bg    = "rgba(255,77,109,0.12)" if is_ai else "rgba(46,196,182,0.12)"
    verdict_border= "#ff4d6d" if is_ai else "#2ec4b6"

    bar_ai_color   = "linear-gradient(90deg,#ff6b6b,#ff4d6d)"
    bar_real_color = "linear-gradient(90deg,#43e8d8,#2ec4b6)"

    return f"""
<div class="result-card">
  <div class="verdict-block" style="background:{verdict_bg};border:1.5px solid {verdict_border};">
    <div class="verdict-icon-wrap" style="color:{verdict_color};">{icon}</div>
    <div class="verdict-text">
      <div class="verdict-label" style="color:{verdict_color};">{verdict}</div>
      <div class="verdict-conf">{conf_text} &mdash; {confidence*100:.1f}%</div>
    </div>
  </div>

  <div class="score-section">
    <div class="score-item">
      <div class="score-header">
        <span class="score-name">🤖 AI-Generated</span>
        <span class="score-pct" style="color:#ff4d6d;">{ai_score*100:.1f}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:{ai_w}%;background:{bar_ai_color};"></div>
      </div>
    </div>

    <div class="score-item" style="margin-top:1.1rem;">
      <div class="score-header">
        <span class="score-name">📷 Human-Created</span>
        <span class="score-pct" style="color:#2ec4b6;">{real_score*100:.1f}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:{real_w}%;background:{bar_real_color};"></div>
      </div>
    </div>
  </div>
</div>
"""


def _empty_state() -> str:
    return """
<div class="empty-state">
  <div class="empty-icon">🔍</div>
  <p>Upload an image and click <strong>Analyze</strong> to see the verdict.</p>
</div>
"""


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@400;500;600&display=swap');

/* ── Reset & root ────────────────────────────────────────────── */
:root {
    --bg:       #0d0f14;
    --surface:  #161a23;
    --surface2: #1e2330;
    --border:   #2a3045;
    --accent:   #7c6af7;
    --accent2:  #a78bfa;
    --text:     #e8eaf0;
    --text-dim: #8b92a9;
    --ai-col:   #ff4d6d;
    --hu-col:   #2ec4b6;
    --shadow:   0 4px 24px rgba(0,0,0,0.45);
}

/* Force body / app background */
body,
.gradio-container,
.gradio-container > *,
footer { background: var(--bg) !important; }

.gradio-container {
    max-width: 980px !important;
    margin: 0 auto !important;
    padding: 1.5rem 1rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

/* ── Header ──────────────────────────────────────────────────── */
#header {
    text-align: center;
    padding: 2.5rem 1rem 2rem;
}
#header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff !important;
    margin: 0 0 0.5rem;
    letter-spacing: -0.03em;
}
#header h1 span { color: var(--accent2); }
#header p {
    font-size: 1rem;
    color: var(--text-dim) !important;
    margin: 0;
}

/* ── Panels ──────────────────────────────────────────────────── */
#main-row {
    gap: 1.25rem !important;
    align-items: stretch !important;
}
#input-panel, #output-panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    box-shadow: var(--shadow) !important;
}

.panel-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--text-dim) !important;
    margin-bottom: 1rem !important;
}

/* ── Upload zone ─────────────────────────────────────────────── */
#image-upload {
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    background: var(--surface2) !important;
    min-height: 260px !important;
    transition: border-color 0.2s !important;
    color: var(--text-dim) !important;
}
#image-upload:hover { border-color: var(--accent) !important; }
#image-upload * { color: var(--text-dim) !important; }

/* ── Buttons ─────────────────────────────────────────────────── */
#btn-analyze {
    background: var(--accent) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    width: 100% !important;
    margin-top: 0.75rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 16px rgba(124,106,247,0.35) !important;
}
#btn-analyze:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

#btn-clear {
    background: var(--surface2) !important;
    color: var(--text-dim) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.7rem 0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    cursor: pointer !important;
    width: 100% !important;
    margin-top: 0.75rem !important;
    transition: border-color 0.2s, color 0.2s !important;
}
#btn-clear:hover {
    border-color: var(--ai-col) !important;
    color: var(--ai-col) !important;
}

/* ── Result card HTML ────────────────────────────────────────── */
.result-card {
    font-family: 'DM Sans', sans-serif;
    color: var(--text, #e8eaf0);
}

.verdict-block {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.1rem 1.25rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.verdict-icon-wrap {
    font-size: 2rem;
    line-height: 1;
    flex-shrink: 0;
}
.verdict-text { display: flex; flex-direction: column; gap: 0.2rem; }
.verdict-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    line-height: 1.2;
}
.verdict-conf {
    font-size: 0.82rem;
    color: #8b92a9;
    font-weight: 500;
}

.score-section { padding: 0 0.1rem; }
.score-item {}
.score-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.45rem;
}
.score-name {
    font-size: 0.9rem;
    font-weight: 500;
    color: #e8eaf0;
}
.score-pct {
    font-size: 0.95rem;
    font-weight: 700;
}
.bar-track {
    width: 100%;
    height: 9px;
    background: #2a3045;
    border-radius: 999px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
}

/* ── Empty / error states ────────────────────────────────────── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 220px;
    color: #8b92a9;
    text-align: center;
    padding: 2rem 1rem;
    font-family: 'DM Sans', sans-serif;
}
.empty-icon { font-size: 2.5rem; margin-bottom: 0.85rem; opacity: 0.4; }
.empty-state p { font-size: 0.93rem; margin: 0; line-height: 1.6; color: #8b92a9; }
.empty-state strong { color: #e8eaf0; }

.result-error {
    font-family: 'DM Sans', sans-serif;
    padding: 1rem 1.25rem;
    background: rgba(255,77,109,0.1);
    border: 1px solid rgba(255,77,109,0.35);
    border-radius: 10px;
    color: #ff8fa3;
    font-size: 0.88rem;
}

/* ── Footer ──────────────────────────────────────────────────── */
#footer {
    text-align: center;
    margin-top: 1.5rem;
    color: #4a5168;
    font-size: 0.78rem;
    font-family: 'DM Sans', sans-serif;
}
#footer strong { color: #6b7599; }

/* ── Gradio overrides – ensure dark surfaces everywhere ─────── */
.svelte-1gfkn6j, .wrap, .gap, .form,
.block, .padded, label, .label-wrap,
.input-container, .output-container {
    background: transparent !important;
    color: var(--text) !important;
}
label span, .label-wrap span {
    color: var(--text-dim) !important;
    font-family: 'DM Sans', sans-serif !important;
}
"""

with gr.Blocks(css=custom_css, title="AI Image Detector") as demo:

    gr.HTML("""
        <div id="header">
            <h1>🔍 AI Image <span>Detector</span></h1>
            <p>Upload any image to detect whether it was AI-generated or human-created.</p>
        </div>
    """)

    with gr.Row(elem_id="main-row", equal_height=True):
        with gr.Column(scale=1, elem_id="input-panel"):
            gr.HTML('<div class="panel-title">Input Image</div>')
            image_input = gr.Image(
                label="", type="filepath", show_label=False, elem_id="image-upload",
            )
            with gr.Row():
                clear_btn  = gr.ClearButton(components=[image_input], value="Clear", elem_id="btn-clear")
                submit_btn = gr.Button(value="Analyze →", variant="primary", elem_id="btn-analyze")

        with gr.Column(scale=1, elem_id="output-panel"):
            gr.HTML('<div class="panel-title">Analysis Result</div>')
            output_html = gr.HTML(value=_empty_state(), elem_id="output-result")

    gr.HTML('<div id="footer">Created by <strong>Arun Kumar</strong> &mdash; VIT Student</div>')

    submit_btn.click(fn=format_output, inputs=[image_input], outputs=[output_html])
    image_input.change(fn=format_output, inputs=[image_input], outputs=[output_html])

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Base(
            primary_hue="violet",
            secondary_hue="slate",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("DM Sans"), "ui-sans-serif", "sans-serif"],
        )
    )