"""
眼底血管抽出 (GPU版 / PyTorch)

従来版からの主な変更点
  1. 解像度正規化   : 眼底の直径を TARGET_DIAMETER に揃えて処理 (画像サイズ依存を排除)
  2. 局所コントラスト正規化 : log輝度 -> 背景除去 -> 局所標準偏差で割る
                       (暗い領域/明るい視神経乳頭周辺でも同じ閾値で扱える)
  3. 方向付きリッジフィルタ束 : 5スケール x 16方向 を GPU で畳み込み
                       (Frangiより太い血管・低コントラスト血管に強い)
  4. エッジ抑制 + 異方性   : 乳頭輪郭/眼底縁/黄斑のブロブ状反応を減衰
  5. ヒステリシス閾値 + スケルトン化 + スパー除去 + 短い成分の除去

使い方:  python vessel_extraction_gpu.py
"""

import os
import shutil

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import remove_small_holes, skeletonize
from tqdm import tqdm

# ============================================================
# 設定
# ============================================================
SOURCE_IMAGE_DIR = "healthy_images"
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

VESSEL_OUTPUT_DIR = "vessel_extracted"
DEBUG_MONTAGE_DIR = "vessel_debug_20"
DEBUG_SAMPLE_COUNT = 20

OUTPUT_IMAGE_MODE = "vessel_only"      # "vessel_only" or "overlay"
OVERLAY_THICKNESS = 3

# --- 解像度 -------------------------------------------------
TARGET_DIAMETER = 1000                 # 眼底直径をこの画素数に揃える
MASK_ERODE_RATIO = 0.02                # 眼底縁から除外する幅 (直径比)

# --- 正規化 -------------------------------------------------
BG_SIGMA = 20.0                        # 背景推定
STD_SIGMA = 25.0                       # 局所std推定
STD_FLOOR_RATIO = 0.80                 # stdの下限 (全体median比) 大きいほどテクスチャを増幅しない

# --- リッジフィルタ -----------------------------------------
SIGMAS = (1.5, 2.1, 2.9, 4.0, 5.2)     # 血管幅 ≒ 2.5*sigma
N_ORIENT = 16
ALONG_FACTOR = 3.2                     # 血管方向のsigma = 3.2*s + 2 (上限12) 長いほどランダムテクスチャに強い
EDGE_WEIGHT = 0.8                      # 大きいほどエッジ(乳頭縁など)を強く抑制
ANISO_POWER = 1.0                      # 大きいほどブロブ(黄斑など)を強く抑制

# --- 閾値 (眼底内の応答分布のパーセンタイル) -----------------
# 細い血管が足りない → LOW/HIGH を下げる
# ノイズ・偽血管が多い → LOW/HIGH を上げる
LOW_PCT = 88.0
HIGH_PCT = 96.5

# --- 後処理 -------------------------------------------------
MIN_HOLE_AREA = 150                    # 小さい穴は埋める (リング状ループ防止)
SPUR_MIN_LEN = 15                      # これより短い枝は除去
MIN_SKELETON_LEN = 60                  # これより短い孤立成分は除去
WEAK_COMP_LEN = 150                    # これより短い成分は応答が強い場合のみ残す
WEAK_COMP_MEAN_FACTOR = 0.9            # 短い成分に要求する平均応答 (HIGH閾値比)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 眼底マスク
# ============================================================
def estimate_fundus_mask(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.5)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(np.clip(0.4 * otsu, 6, 25))
    mask = (gray > thr).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask), 1.0
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = np.where(lab == largest, 255, 0).astype(np.uint8)
    mask = (ndi.binary_fill_holes(mask > 0)).astype(np.uint8) * 255

    ys, xs = np.nonzero(mask)
    diameter = float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
    return mask, diameter


def erode_mask(mask, width):
    width = max(1, int(round(width)))
    size = 2 * width + 1
    pad = size
    padded = cv2.copyMakeBorder(mask, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    out = cv2.erode(padded, kernel)
    return out[pad:pad + mask.shape[0], pad:pad + mask.shape[1]]


# ============================================================
# 前処理: log -> 背景除去 -> 局所コントラスト正規化
# ============================================================
def masked_blur(img, mask_f, sigma):
    num = cv2.GaussianBlur(img * mask_f, (0, 0), sigma)
    den = cv2.GaussianBlur(mask_f, (0, 0), sigma)
    return num / np.maximum(den, 1e-3)


def normalize_local_contrast(green, mask):
    m = (mask > 0).astype(np.float32)
    L = np.log(green.astype(np.float32) + 10.0)

    # マスク外を周囲の平均で外挿 (境界に偽エッジを作らない)
    ext = masked_blur(L, m, 30.0)
    filled = np.where(m > 0, L, ext).astype(np.float32)

    bg = cv2.GaussianBlur(filled, (0, 0), BG_SIGMA)
    resid = filled - bg

    var = masked_blur(resid * resid, m, STD_SIGMA)
    std = np.sqrt(np.maximum(var, 0))
    valid_std = std[m > 0]
    floor = STD_FLOOR_RATIO * (np.median(valid_std) if valid_std.size else 1.0)
    std = np.maximum(std, max(floor, 1e-4))

    norm = resid / std
    norm[m == 0] = 0.0
    return norm.astype(np.float32)


# ============================================================
# GPU: 方向付きリッジ / エッジフィルタ束
# ============================================================
def build_kernels(sigma, n_orient):
    sl = min(ALONG_FACTOR * sigma + 2.0, 12.0)
    half = int(np.ceil(3.0 * max(sigma, sl)))
    ax = np.arange(-half, half + 1, dtype=np.float32)
    yy, xx = np.meshgrid(ax, ax, indexing="ij")

    ridges, edges = [], []
    for i in range(n_orient):
        th = np.pi * i / n_orient
        u = xx * np.cos(th) + yy * np.sin(th)       # 血管に沿う方向
        v = -xx * np.sin(th) + yy * np.cos(th)      # 血管を横切る方向
        g = np.exp(-(u ** 2) / (2 * sl ** 2) - (v ** 2) / (2 * sigma ** 2))
        g = g / g.sum()

        ridge = (v ** 2 / sigma ** 2 - 1.0) * g     # スケール正規化2階微分
        ridge = ridge - g * (ridge.sum() / g.sum()) # ゼロ平均化
        edge = -(v / sigma) * g                     # スケール正規化1階微分
        ridges.append(ridge)
        edges.append(edge)

    kernels = np.stack(ridges + edges, axis=0)[:, None]  # [2*O,1,k,k]
    return torch.from_numpy(kernels.astype(np.float32)), half


@torch.no_grad()
def ridge_response_gpu(norm_img):
    x = torch.from_numpy(norm_img)[None, None].to(DEVICE)
    best = None

    for sigma in SIGMAS:
        kernels, half = build_kernels(sigma, N_ORIENT)
        kernels = kernels.to(DEVICE)
        xp = F.pad(x, (half, half, half, half), mode="reflect")
        out = F.conv2d(xp, kernels)                  # [1,2*O,H,W]

        R = out[:, :N_ORIENT].clamp_min(0)
        E = out[:, N_ORIENT:].abs()

        r_max, idx = R.max(dim=1, keepdim=True)
        e_at = E.gather(1, idx)
        r_mean = R.mean(dim=1, keepdim=True)

        # 線状 (1方向だけ強い) ほど1、ブロブ (全方向で反応) ほど0
        aniso = (1.0 - r_mean / (r_max + 1e-6)).clamp(0, 1) ** ANISO_POWER
        score = (r_max - EDGE_WEIGHT * e_at).clamp_min(0) * aniso

        best = score if best is None else torch.maximum(best, score)
        del out, R, E, xp

    resp = best[0, 0].float().cpu().numpy()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return resp


# ============================================================
# 後処理
# ============================================================
_NB_KERNEL = np.ones((3, 3), np.float32)
_NB_KERNEL[1, 1] = 0


def neighbor_count(sk):
    return cv2.filter2D(sk.astype(np.float32), -1, _NB_KERNEL,
                        borderType=cv2.BORDER_CONSTANT).astype(np.int32)


def prune_spurs(skel, min_len, iterations=2):
    skel = skel.astype(bool)
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(iterations):
        sk = skel.astype(np.uint8)
        nb = neighbor_count(sk)
        branch = (sk > 0) & (nb >= 3)
        endp = (sk > 0) & (nb == 1)
        branch_d = cv2.dilate(branch.astype(np.uint8), k3) > 0
        segs = (sk > 0) & ~branch_d

        n, lab, stats, _ = cv2.connectedComponentsWithStats(
            segs.astype(np.uint8), connectivity=8)
        if n <= 1:
            break
        near_branch = cv2.dilate(branch_d.astype(np.uint8), k3) > 0
        end_labels = np.unique(lab[endp & segs])
        adj_labels = np.unique(lab[near_branch & segs])
        spur = np.intersect1d(end_labels, adj_labels)
        spur = [l for l in spur if l > 0 and stats[l, cv2.CC_STAT_AREA] < min_len]
        if not spur:
            break
        skel = skel & ~np.isin(lab, spur)
        skel = skeletonize(skel)
    return skel


def remove_short_skeletons(skel, min_len, resp=None, strong_thr=None):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        skel.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_len

    # 短い成分は平均応答が弱ければノイズとみなして除去
    if resp is not None and strong_thr is not None:
        for l in range(1, n):
            if not keep[l]:
                continue
            length = stats[l, cv2.CC_STAT_AREA]
            if length < WEAK_COMP_LEN:
                mean_r = float(resp[lab == l].mean())
                if mean_r < WEAK_COMP_MEAN_FACTOR * strong_thr:
                    keep[l] = False
    return keep[lab]


def create_overlay(original, skeleton, thickness):
    line = skeleton.astype(np.uint8) * 255
    if thickness > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        line = cv2.dilate(line, k)
    out = original.copy()
    out[line > 0] = (0, 0, 255)
    return out


# ============================================================
# 1枚処理
# ============================================================
def extract_vessels(original):
    h0, w0 = original.shape[:2]
    mask0, diameter = estimate_fundus_mask(original)
    if np.count_nonzero(mask0) == 0:
        raise RuntimeError("眼底領域を検出できません")

    # ---- 作業解像度へ -----------------------------------
    scale = TARGET_DIAMETER / diameter
    if abs(scale - 1.0) > 0.03:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        img = cv2.resize(original, None, fx=scale, fy=scale, interpolation=interp)
        mask = cv2.resize(mask0, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    else:
        scale = 1.0
        img, mask = original, mask0

    mask_eff = erode_mask(mask, MASK_ERODE_RATIO * TARGET_DIAMETER)
    if np.count_nonzero(mask_eff) == 0:
        mask_eff = mask.copy()

    # ---- 前処理 -----------------------------------------
    green = img[:, :, 1]
    norm = normalize_local_contrast(green, mask_eff)

    # ---- GPU フィルタ -----------------------------------
    resp = ridge_response_gpu(norm)
    resp[mask_eff == 0] = 0.0
    vals = resp[mask_eff > 0]
    ref = np.percentile(vals, 99.0) if vals.size else 1.0
    resp = resp / max(ref, 1e-6)

    # ---- 二値化 -----------------------------------------
    low = np.percentile(resp[mask_eff > 0], LOW_PCT)
    high = np.percentile(resp[mask_eff > 0], HIGH_PCT)
    binary = apply_hysteresis_threshold(resp, low, high) & (mask_eff > 0)
    binary = remove_small_holes(binary, MIN_HOLE_AREA)
    binary &= mask_eff > 0

    # ---- 細線化・整形 -----------------------------------
    skel = skeletonize(binary)
    skel = prune_spurs(skel, SPUR_MIN_LEN)
    skel = remove_short_skeletons(skel, MIN_SKELETON_LEN, resp, high)

    # ---- 元解像度へ戻す ---------------------------------
    if scale != 1.0:
        s8 = cv2.dilate(skel.astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
        s8 = cv2.resize(s8, (w0, h0), interpolation=cv2.INTER_LINEAR)
        skel0 = skeletonize(s8 > 127)
    else:
        skel0 = skel
    skel0 &= erode_mask(mask0, 2) > 0

    debug = {
        "original": original,
        "mask": mask0,
        "normalized": norm,
        "ridge response": resp,
        "binary": binary.astype(np.uint8) * 255,
        "overlay": create_overlay(original, skel0, 4),
    }
    return skel0, debug


# ============================================================
# デバッグモンタージュ
# ============================================================
def to_bgr_u8(img):
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        lo, hi = np.percentile(img, 1), np.percentile(img, 99)
        img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
        img = (img * 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def build_montage(debug, tile_w=560, cols=3):
    tiles = []
    for title, img in debug.items():
        v = to_bgr_u8(img)
        s = tile_w / v.shape[1]
        v = cv2.resize(v, (tile_w, int(v.shape[0] * s)), interpolation=cv2.INTER_AREA)
        cv2.rectangle(v, (0, 0), (v.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(v, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(v)
    th = max(t.shape[0] for t in tiles)
    rows = int(np.ceil(len(tiles) / cols))
    canvas = np.zeros((rows * th, cols * tile_w, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * th:r * th + t.shape[0], c * tile_w:(c + 1) * tile_w] = t
    return canvas


def clear_dir(d):
    os.makedirs(d, exist_ok=True)
    for name in os.listdir(d):
        p = os.path.join(d, name)
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)


# ============================================================
# main
# ============================================================
def main():
    if not os.path.isdir(SOURCE_IMAGE_DIR):
        raise FileNotFoundError(f"入力フォルダがありません: {SOURCE_IMAGE_DIR}")

    files = sorted(
        f for f in os.listdir(SOURCE_IMAGE_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError("対応画像がありません。")

    clear_dir(VESSEL_OUTPUT_DIR)
    clear_dir(DEBUG_MONTAGE_DIR)
    print(f"device: {DEVICE}  /  images: {len(files)}")

    ok = 0
    for i, name in enumerate(tqdm(files, desc="Vessel extraction")):
        img = cv2.imread(os.path.join(SOURCE_IMAGE_DIR, name), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[SKIP] 読み込み失敗: {name}")
            continue
        try:
            skel, debug = extract_vessels(img)
        except Exception as exc:
            print(f"[SKIP] {name}: {type(exc).__name__}: {exc}")
            continue

        stem = os.path.splitext(name)[0]
        if OUTPUT_IMAGE_MODE == "overlay":
            out = create_overlay(img, skel, OVERLAY_THICKNESS)
        else:
            out = skel.astype(np.uint8) * 255
        cv2.imwrite(os.path.join(VESSEL_OUTPUT_DIR, stem + "_vessel.png"), out)

        if i < DEBUG_SAMPLE_COUNT:
            cv2.imwrite(
                os.path.join(DEBUG_MONTAGE_DIR, f"{i + 1:02d}_{stem}_debug.jpg"),
                build_montage(debug), [cv2.IMWRITE_JPEG_QUALITY, 92])
        ok += 1

    print(f"完了: {ok}/{len(files)} 枚 -> {VESSEL_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
