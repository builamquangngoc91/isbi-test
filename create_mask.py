import os, json, pickle, cv2, torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from scipy.ndimage import gaussian_filter

# ---------------- CONFIG ---------------- #
HF_TOKEN = ""
MODEL_REPO = "MahmoodLab/CONCH"
MODEL_NAME = "conch_ViT-B-16"
PATCH_SIZE = 16
VAL_DIR = r".\BCSS-WSSS\training"
prototype_path = r"class_patch_prototypes_with_bg_refined.pkl"
json_file = r"masked_dataset.json"
SAVE_INTERVAL = 20  # Save JSON sau mỗi 20 ảnh

classes = ["Tumor", "Stroma", "Lymphocytic infiltrate", "Necrosis", "Background"]

# ---------------- LOAD MODEL ---------------- #
from conch.open_clip_custom import create_model_from_pretrained
device = "cpu"

print("🧠 Loading model...")
model, preprocess = create_model_from_pretrained(
    MODEL_NAME,
    f"hf_hub:{MODEL_REPO}",
    hf_auth_token=HF_TOKEN
)
model = model.eval().to(device)

# ---------------- LOAD PROTOTYPES ---------------- #
print("📦 Loading prototypes...")
with open(prototype_path, "rb") as f:
    prototype_embs = pickle.load(f)

proto_tensor = {}
for cls, arrs in prototype_embs.items():
    arr = np.stack(arrs, axis=0)
    t = torch.from_numpy(arr).float().to(device)
    proto_tensor[cls] = F.normalize(t.mean(dim=0, keepdim=True), p=2, dim=-1)
print("✅ Loaded prototypes:", list(proto_tensor.keys()))

# ---------------- PREDICT FUNCTION ---------------- #
def predict_label_map_smooth(img_path, model, proto_tensor, preprocess, patch_size=PATCH_SIZE, sigma=1.0, batch_size=64):
    """Sinh bản đồ nhãn mượt mà không resize"""
    all_classes = list(proto_tensor.keys())

    img_pil = Image.open(img_path).convert("RGB")
    orig = np.array(img_pil)
    H, W = orig.shape[:2]

    to_tensor = transforms.ToTensor()
    img_t = to_tensor(img_pil).unsqueeze(0).to(device)
    _, _, Ht, Wt = img_t.shape

    # patchify (không resize)
    n_h = Ht // patch_size
    n_w = Wt // patch_size
    img_t = img_t[:, :, :n_h * patch_size, :n_w * patch_size]
    patches = img_t.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, patch_size, patch_size)

    all_embs = []
    with torch.inference_mode():
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i + batch_size]
            emb = model.encode_image(batch, proj_contrast=True, normalize=True)
            emb = F.normalize(emb, dim=-1)
            all_embs.append(emb)
    patch_embs = torch.cat(all_embs, dim=0)

    sims = []
    for cls in all_classes:
        proto = F.normalize(proto_tensor[cls].to(device), dim=-1)
        s = (patch_embs @ proto.T).squeeze(1)
        sims.append(s.unsqueeze(1))
    sims = torch.cat(sims, dim=1)

    sims_map = sims.cpu().numpy().reshape(n_h, n_w, len(all_classes))

    # smoothing
    for c in range(len(all_classes)):
        sims_map[..., c] = gaussian_filter(sims_map[..., c], sigma=sigma)

    # upsample
    sims_map_up = np.zeros((Ht, Wt, len(all_classes)), dtype=np.float32)
    for c in range(len(all_classes)):
        sims_map_up[..., c] = cv2.resize(sims_map[..., c], (Wt, Ht), interpolation=cv2.INTER_LINEAR)

    pred_map = sims_map_up.argmax(axis=-1)
    pred_map_full = cv2.resize(pred_map, (W, H), interpolation=cv2.INTER_NEAREST)
    return pred_map_full

# ---------------- MASK PER CLASS ---------------- #
def mask_image_per_class(orig_img, pred_map, border=1):
    """Tạo mask riêng cho từng class, chỉ chừa viền."""
    masked_images = {}
    num_classes = pred_map.max() + 1

    for cls in range(num_classes):
        cls_mask = (pred_map == cls).astype(np.uint8)
        if cls_mask.sum() == 0:
            continue  # bỏ class không có pixel

        masked_img = np.ones_like(pred_map, dtype=np.uint8)
        contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            region_mask = np.zeros_like(cls_mask)
            cv2.drawContours(region_mask, [cnt], -1, 1, thickness=-1)
            if border > 0:
                region_mask = cv2.erode(region_mask, np.ones((border * 2 + 1, border * 2 + 1), np.uint8))
            masked_img[region_mask == 1] = 0

        masked_images[cls] = masked_img

    return masked_images

# ---------------- SAFE JSON LOAD ---------------- #
def load_json_safe(json_file):
    if not os.path.exists(json_file):
        return {}
    try:
        with open(json_file, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON error at pos {e.pos}. Attempting partial recovery...")
        with open(json_file, "r") as f:
            data = f.read()
        for cut in range(100000, len(data), 100000):
            try:
                partial = data[:-cut]
                recovered = json.loads(partial + "}")
                print(f"✅ Recovered partial JSON with {len(recovered)} entries.")
                return recovered
            except json.JSONDecodeError:
                continue
        print("⚠️ Could not recover any valid JSON. Starting fresh.")
        return {}

# ---------------- MAIN ---------------- #
os.makedirs(os.path.dirname(json_file) or ".", exist_ok=True)

all_masks = load_json_safe(json_file)
print(f"📊 Loaded {len(all_masks)} existing masks from JSON.")

val_images = [os.path.join(VAL_DIR, f) for f in sorted(os.listdir(VAL_DIR)) if f.endswith(".png")]
remaining = [p for p in val_images if os.path.basename(p) not in all_masks]

print(f"🖼️ Total images: {len(val_images)} | ✅ Done: {len(all_masks)} | ⏳ Remaining: {len(remaining)}")

for i, img_path in enumerate(tqdm(remaining, desc="Processing images")):
    img_name = os.path.basename(img_path)

    try:
        pred = predict_label_map_smooth(img_path, model, proto_tensor, preprocess, PATCH_SIZE, sigma=1.0, batch_size=32)
        masked_dict = mask_image_per_class(np.array(Image.open(img_path).convert("RGB")), pred, border=1)

        # chỉ lưu nếu có ít nhất 1 region
        if len(masked_dict) == 0:
            print(f"⚠️ {img_name}: không có class nào, bỏ qua.")
            continue

        # chuyển về list để JSON hóa
        masked_list = [masked_dict[cls].tolist() for cls in sorted(masked_dict.keys())]
        all_masks[img_name] = masked_list

    except Exception as e:
        print(f"⚠️ Error processing {img_name}: {e}")
        continue

    # chỉ ghi khi xử lý xong 1 ảnh (đủ region) và mỗi SAVE_INTERVAL
    if (i + 1) % SAVE_INTERVAL == 0 or i == len(remaining) - 1:
        tmp_file = json_file + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(all_masks, f)
        os.replace(tmp_file, json_file)
        print(f"💾 Saved progress safely ({len(all_masks)} masks so far)")

print(f"✅ Done! Final total: {len(all_masks)} masks saved to {json_file}")
