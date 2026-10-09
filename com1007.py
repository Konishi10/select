"""
眼底血管の複雑性ランキング

入力:
    healthy_images/
        元の眼底画像

    vessel_extracted/
        xxx_vessel.png

出力:
    vessel_complexity_results.csv

    complexity_top20/
        Top10のデバッグ画像

    complexity_graphs/
        複雑性ランキングの根拠グラフ


ランキング:
    Fractal Dimension (FD) 70% と分岐数 30% を用いた
    複雑さスコアでランキングする。

分岐点:
    Cross Number (CN) を用いて分岐候補を検出する。

    ただし、
        「CN >= 3 の画素を8近傍クラスタリングして
         そのcluster数を分岐数とする」
    だけではなく、

        1. CN >= 3 の候補領域を検出
        2. 候補領域の外側へジャンプ
        3. 外側のスケルトンを追跡
        4. 独立した血管枝の本数を確認
        5. 3本以上の枝が出ている候補だけを
           1つの実際の分岐として数える

    方式を使用する。

skeleton_length はランキングスコアには使用せず、
分岐数とFDのみをランキングに使用する。
"""


import os
import shutil
from collections import deque

import cv2
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tqdm import tqdm


# ============================================================
# 基本設定
# ============================================================

SOURCE_IMAGE_DIR = "healthy_images"
VESSEL_DIR = "vessel_extracted"

OUTPUT_CSV = "vessel_complexity_results.csv"

TOP20_DIR = "complexity_top10"
GRAPH_DIR = "complexity_graphs"

TOP_K = 10

SUPPORTED_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
)


# ============================================================
# 分岐追跡設定
# ============================================================

# CN >= 3 を分岐候補とする
BRANCH_CN_THRESHOLD = 3

# 分岐候補から外側の血管へ何画素程度ジャンプするか
#
# 小さすぎる:
#   分岐候補領域そのものを何度も拾いやすい
#
# 大きすぎる:
#   近接した別枝を飛び越える可能性がある
#
# 通常のスケルトンでは 2～4 程度から開始する。
BRANCH_JUMP_DISTANCE = 3

# 分岐候補の周辺で、独立した枝とみなすための
# 最小追跡距離
BRANCH_TRACE_MIN_LENGTH = 4

# 同じ分岐から出た枝同士の方向が近い場合、
# 同じ枝として扱う角度閾値
#
# 例:
#   20度以内 → 同じ枝候補
BRANCH_DIRECTION_ANGLE = 25.0

# 同じ分岐とみなす候補中心間距離
# 近接するCN候補が同じ分岐を表す場合の重複除去用
BRANCH_MERGE_DISTANCE = 12


# ============================================================
# 出力フォルダ初期化
# ============================================================

def clear_dir(directory):

    os.makedirs(
        directory,
        exist_ok=True
    )

    for name in os.listdir(directory):

        path = os.path.join(
            directory,
            name
        )

        if os.path.isdir(path):
            shutil.rmtree(path)

        else:
            os.remove(path)


# ============================================================
# ファイル名
# ============================================================

def get_stem(filename):

    return os.path.splitext(
        os.path.basename(filename)
    )[0]


def build_original_image_map():

    """
    healthy_images 内の画像を

        stem -> filepath

    にする。
    """

    image_map = {}

    for filename in os.listdir(
        SOURCE_IMAGE_DIR
    ):

        ext = os.path.splitext(
            filename
        )[1].lower()

        if ext not in SUPPORTED_EXTENSIONS:
            continue

        stem = get_stem(
            filename
        )

        image_map[stem] = os.path.join(
            SOURCE_IMAGE_DIR,
            filename
        )

    return image_map


def get_image_id_from_vessel(filename):

    """
    xxx_vessel.png
        ↓
    xxx
    """

    stem = get_stem(
        filename
    )

    suffix = "_vessel"

    if stem.endswith(suffix):

        return stem[
            :-len(suffix)
        ]

    return stem


# ============================================================
# Skeleton判定
# ============================================================

def make_binary_skeleton(image):

    """
    vessel画像を0/1のスケルトンにする。
    """

    if image is None:
        return None

    skeleton = (
        image > 0
    ).astype(
        np.uint8
    )

    return skeleton


# ============================================================
# 8近傍
# ============================================================

NEIGHBOR_OFFSETS = [

    (-1, -1),
    (-1,  0),
    (-1,  1),

    ( 0, -1),
    ( 0,  1),

    ( 1, -1),
    ( 1,  0),
    ( 1,  1)

]


# ============================================================
# Cross Number
# ============================================================

def crossing_number_at(
    skeleton,
    y,
    x
):

    """
    8近傍のCrossing Numberを計算する。

          p0 p1 p2
          p7  x p3
          p6 p5 p4

    CN = 1/2 * Σ |p_i - p_(i+1)|

    目安:

        CN = 0 -> 孤立点
        CN = 1 -> endpoint
        CN = 2 -> 通常の血管
        CN >= 3 -> 分岐候補
    """

    h, w = skeleton.shape

    if (
        y <= 0
        or y >= h - 1
        or x <= 0
        or x >= w - 1
    ):
        return 0

    p = [

        skeleton[y - 1, x],

        skeleton[y - 1, x + 1],

        skeleton[y, x + 1],

        skeleton[y + 1, x + 1],

        skeleton[y + 1, x],

        skeleton[y + 1, x - 1],

        skeleton[y, x - 1],

        skeleton[y - 1, x - 1]
    ]

    transitions = 0

    for i in range(8):

        transitions += abs(
            int(p[i])
            - int(p[(i + 1) % 8])
        )

    return transitions // 2


# ============================================================
# CN画像
# ============================================================

def calculate_cn_image(
    skeleton
):

    """
    各スケルトン画素のCross Numberを計算する。
    """

    skeleton = (
        skeleton > 0
    ).astype(
        np.uint8
    )

    cn_image = np.zeros_like(
        skeleton,
        dtype=np.uint8
    )

    ys, xs = np.where(
        skeleton > 0
    )

    for y, x in zip(
        ys,
        xs
    ):

        cn_image[y, x] = crossing_number_at(
            skeleton,
            int(y),
            int(x)
        )

    return cn_image


# ============================================================
# 分岐候補
# ============================================================

def detect_branch_candidates(
    skeleton
):

    """
    CN >= 3 の画素を分岐候補とする。
    """

    cn_image = calculate_cn_image(
        skeleton
    )

    branch_mask = (
        cn_image >= BRANCH_CN_THRESHOLD
    ).astype(
        np.uint8
    )

    return (
        branch_mask,
        cn_image
    )


# ============================================================
# 8近傍で周囲画素取得
# ============================================================

def get_neighbors(
    skeleton,
    y,
    x
):

    h, w = skeleton.shape

    neighbors = []

    for dy, dx in NEIGHBOR_OFFSETS:

        ny = y + dy
        nx = x + dx

        if (
            0 <= ny < h
            and 0 <= nx < w
            and skeleton[ny, nx] > 0
        ):

            neighbors.append(
                (ny, nx)
            )

    return neighbors


# ============================================================
# 候補画素から「外側」にジャンプ
# ============================================================

def jump_from_candidate(
    skeleton,
    branch_mask,
    start_y,
    start_x,
    direction_y,
    direction_x,
    jump_distance
):

    """
    分岐候補領域の内部を歩き続けるのではなく、
    指定方向へジャンプして、
    分岐候補領域の外側にある血管画素を探す。

    戻り値:
        (y, x)

    見つからない場合:
        None
    """

    h, w = skeleton.shape

    y = float(start_y)
    x = float(start_x)

    for _ in range(
        jump_distance
    ):

        y += direction_y
        x += direction_x

    cy = int(
        round(y)
    )

    cx = int(
        round(x)
    )

    search_radius = 2

    best = None
    best_distance = None

    for dy in range(
        -search_radius,
        search_radius + 1
    ):

        for dx in range(
            -search_radius,
            search_radius + 1
        ):

            ny = cy + dy
            nx = cx + dx

            if (
                ny < 0
                or ny >= h
                or nx < 0
                or nx >= w
            ):
                continue

            if skeleton[ny, nx] == 0:
                continue

            # 分岐候補内部なら、
            # 「外側へのジャンプ」という目的に
            # 合わないので除外
            if branch_mask[ny, nx] > 0:
                continue

            distance = (
                dy * dy
                + dx * dx
            )

            if (
                best is None
                or distance < best_distance
            ):

                best = (
                    ny,
                    nx
                )

                best_distance = distance

    return best


# ============================================================
# 候補領域の周囲から外側の枝を探す
# ============================================================

def find_outgoing_branches(
    skeleton,
    branch_mask,
    component_mask,
    centroid_y,
    centroid_x
):

    """
    分岐候補から外側へジャンプし、
    独立した血管枝を取得する。

    「候補画素の数」を数えるのではなく、

        分岐候補
             ↓
        外側へジャンプ
             ↓
        血管を追跡

    という処理を行う。
    """

    h, w = skeleton.shape

    candidate_pixels = np.argwhere(
        component_mask > 0
    )

    if len(candidate_pixels) == 0:
        return []

    jump_points = []

    # --------------------------------------------------------
    # 候補領域の各画素から、
    # 8方向へジャンプ
    # --------------------------------------------------------

    for y, x in candidate_pixels:

        y = int(y)
        x = int(x)

        dy = float(
            y - centroid_y
        )

        dx = float(
            x - centroid_x
        )

        norm = np.sqrt(
            dx * dx
            + dy * dy
        )

        # 中心と同じ位置の場合は、
        # 8方向すべてを調べる
        if norm < 1e-6:

            directions = [

                (-1, -1),
                (-1,  0),
                (-1,  1),

                ( 0, -1),
                ( 0,  1),

                ( 1, -1),
                ( 1,  0),
                ( 1,  1)
            ]

        else:

            base_dy = dy / norm
            base_dx = dx / norm

            directions = [

                (
                    base_dy,
                    base_dx
                )
            ]

        for direction_y, direction_x in directions:

            point = jump_from_candidate(
                skeleton,
                branch_mask,
                y,
                x,
                direction_y,
                direction_x,
                BRANCH_JUMP_DISTANCE
            )

            if point is None:
                continue

            py, px = point

            # ------------------------------------------------
            # 候補領域外に出ていること
            # ------------------------------------------------

            if component_mask[py, px] > 0:
                continue

            jump_points.append(
                (py, px)
            )

    # --------------------------------------------------------
    # 重複除去
    # --------------------------------------------------------

    unique_points = []

    for point in jump_points:

        if point in unique_points:
            continue

        unique_points.append(
            point
        )

    if not unique_points:
        return []

    # --------------------------------------------------------
    # 各ジャンプ地点から血管を追跡
    # --------------------------------------------------------

    branches = []

    visited_global = set()

    for start in unique_points:

        if start in visited_global:
            continue

        trace_result = trace_vessel_branch(
            skeleton,
            start,
            component_mask,
            max_steps=5000
        )

        if trace_result is None:
            continue

        path, end_reason = trace_result

        if len(path) < BRANCH_TRACE_MIN_LENGTH:
            continue

        for point in path:
            visited_global.add(point)

        branches.append({

            "start": start,

            "path": path,

            "end_reason": end_reason
        })

    # --------------------------------------------------------
    # 方向の近い枝を統合
    # --------------------------------------------------------

    merged_branches = merge_similar_branches(
        branches
    )

    return merged_branches


# ============================================================
# 血管を追跡
# ============================================================

def trace_vessel_branch(
    skeleton,
    start,
    component_mask,
    max_steps=5000
):

    """
    startからスケルトンをたどる。

    分岐候補領域を抜けたあと、

        endpoint
        または
        次の分岐

    に到達するまで追跡する。

    戻り値:
        path
        end_reason
    """

    h, w = skeleton.shape

    current = start

    previous = None

    path = []

    visited = set()

    for _ in range(
        max_steps
    ):

        y, x = current

        if (
            y < 0
            or y >= h
            or x < 0
            or x >= w
        ):
            return (
                path,
                "outside"
            )

        if current in visited:
            return (
                path,
                "loop"
            )

        visited.add(
            current
        )

        path.append(
            current
        )

        # ----------------------------------------------------
        # 現在位置のCN
        # ----------------------------------------------------

        cn = crossing_number_at(
            skeleton,
            y,
            x
        )

        # 自分自身の分岐候補領域に戻った場合
        if (
            len(path) > 1
            and component_mask[y, x] > 0
        ):

            return (
                path,
                "junction"
            )

        # ----------------------------------------------------
        # 周囲の血管
        # ----------------------------------------------------

        neighbors = get_neighbors(
            skeleton,
            y,
            x
        )

        # 前の画素を除く
        if previous is not None:

            neighbors = [
                p
                for p in neighbors
                if p != previous
            ]

        # ----------------------------------------------------
        # 次の分岐
        # ----------------------------------------------------

        if (
            len(path) > BRANCH_TRACE_MIN_LENGTH
            and cn >= BRANCH_CN_THRESHOLD
        ):

            return (
                path,
                "next_branch"
            )

        # ----------------------------------------------------
        # endpoint
        # ----------------------------------------------------

        if len(neighbors) == 0:

            return (
                path,
                "endpoint"
            )

        # ----------------------------------------------------
        # ループ
        # ----------------------------------------------------

        if len(neighbors) > 1:

            # 分岐候補でない場合は、
            # 多少のスケルトン分岐ノイズとして
            # 最も直進方向に近いものを選択する。
            next_point = choose_straightest_neighbor(
                previous,
                current,
                neighbors
            )

        else:

            next_point = neighbors[0]

        previous = current
        current = next_point

    return (
        path,
        "max_steps"
    )


# ============================================================
# 最も直進方向に近い枝を選ぶ
# ============================================================

def choose_straightest_neighbor(
    previous,
    current,
    neighbors
):

    """
    前進方向を維持する候補を選ぶ。

    これにより、細かいスケルトンの揺れで
    追跡方向が突然変わることを抑える。
    """

    if previous is None:
        return neighbors[0]

    py, px = previous
    cy, cx = current

    vx = cx - px
    vy = cy - py

    vnorm = np.sqrt(
        vx * vx
        + vy * vy
    )

    if vnorm < 1e-6:
        return neighbors[0]

    vx /= vnorm
    vy /= vnorm

    best = neighbors[0]
    best_score = -999999.0

    for ny, nx in neighbors:

        dx = nx - cx
        dy = ny - cy

        norm = np.sqrt(
            dx * dx
            + dy * dy
        )

        if norm < 1e-6:
            continue

        dx /= norm
        dy /= norm

        score = (
            vx * dx
            + vy * dy
        )

        if score > best_score:

            best_score = score
            best = (
                ny,
                nx
            )

    return best


# ============================================================
# 枝方向
# ============================================================

def calculate_branch_direction(
    branch
):

    path = branch["path"]

    if len(path) < 2:
        return None

    # 最初の数画素ではなく、
    # 少し進んだ位置から方向を取る。
    index = min(
        5,
        len(path) - 1
    )

    y0, x0 = path[0]
    y1, x1 = path[index]

    dx = x1 - x0
    dy = y1 - y0

    norm = np.sqrt(
        dx * dx
        + dy * dy
    )

    if norm < 1e-6:
        return None

    return (
        dx / norm,
        dy / norm
    )


# ============================================================
# 角度差
# ============================================================

def angle_between(
    direction1,
    direction2
):

    if (
        direction1 is None
        or direction2 is None
    ):
        return 180.0

    dx1, dy1 = direction1
    dx2, dy2 = direction2

    dot = (
        dx1 * dx2
        + dy1 * dy2
    )

    dot = np.clip(
        dot,
        -1.0,
        1.0
    )

    return float(
        np.degrees(
            np.arccos(dot)
        )
    )


# ============================================================
# 似た方向の枝を統合
# ============================================================

def merge_similar_branches(
    branches
):

    """
    同じ血管枝について、ジャンプ地点が
    複数発生する場合がある。

    方向が近いものを同一枝として統合する。
    """

    if not branches:
        return []

    merged = []

    for branch in branches:

        direction = calculate_branch_direction(
            branch
        )

        branch["direction"] = direction

        assigned = False

        for existing in merged:

            angle = angle_between(
                direction,
                existing["direction"]
            )

            if angle <= BRANCH_DIRECTION_ANGLE:

                # より長い追跡結果を残す
                if len(branch["path"]) > len(
                    existing["path"]
                ):

                    existing["path"] = branch[
                        "path"
                    ]

                    existing["end_reason"] = (
                        branch["end_reason"]
                    )

                assigned = True
                break

        if not assigned:

            merged.append(
                branch
            )

    return merged


# ============================================================
# 分岐中心候補
# ============================================================

def find_branch_components(
    branch_mask
):

    """
    CN>=3領域を候補領域として取得する。

    ここでのconnected componentは、
    最終的なbranch_pointsそのものではない。

    後段で「実際に3本以上の枝が出ているか」を
    追跡して確認する。
    """

    if np.sum(branch_mask) == 0:

        return (
            0,
            np.zeros_like(
                branch_mask,
                dtype=np.int32
            )
        )

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            branch_mask.astype(
                np.uint8
            ),
            connectivity=8
        )
    )

    return (
        num_labels - 1,
        labels
    )


# ============================================================
# 分岐点計算
# ============================================================

def count_branch_points(
    skeleton,
    return_debug=False
):

    """
    分岐数を追跡方式で計算する。

    STEP 1:
        CN >= 3 の候補を検出

    STEP 2:
        候補領域を取得

    STEP 3:
        候補領域から外側へジャンプ

    STEP 4:
        ジャンプ地点から血管を追跡

    STEP 5:
        独立した枝が3本以上あれば
        1つの実際の分岐として認定

    STEP 6:
        近接した同一分岐候補の重複を除去
    """

    skeleton = (
        skeleton > 0
    ).astype(
        np.uint8
    )

    branch_mask, cn_image = (
        detect_branch_candidates(
            skeleton
        )
    )

    num_components, labels = (
        find_branch_components(
            branch_mask
        )
    )

    accepted = []

    # --------------------------------------------------------
    # 各候補について枝を追跡
    # --------------------------------------------------------

    for label in range(
        1,
        num_components + 1
    ):

        component_mask = (
            labels == label
        ).astype(
            np.uint8
        )

        ys, xs = np.where(
            component_mask > 0
        )

        if len(xs) == 0:
            continue

        centroid_x = float(
            np.mean(xs)
        )

        centroid_y = float(
            np.mean(ys)
        )

        branches = find_outgoing_branches(
            skeleton,
            branch_mask,
            component_mask,
            centroid_y,
            centroid_x
        )

        # ----------------------------------------------------
        # 3本以上の独立枝が確認できた場合だけ
        # 本当の分岐とする
        # ----------------------------------------------------

        if len(branches) >= 3:

            accepted.append({

                "label": label,

                "x": centroid_x,

                "y": centroid_y,

                "branches": branches,

                "branch_count": len(branches)
            })

    # --------------------------------------------------------
    # 近接する同一分岐候補を統合
    # --------------------------------------------------------

    final_branches = []

    for branch in accepted:

        x = branch["x"]
        y = branch["y"]

        merged = False

        for existing in final_branches:

            distance = np.sqrt(
                (
                    x
                    - existing["x"]
                ) ** 2
                +
                (
                    y
                    - existing["y"]
                ) ** 2
            )

            if distance <= BRANCH_MERGE_DISTANCE:

                # より多くの枝を確認できた候補を残す
                if (
                    branch["branch_count"]
                    >
                    existing["branch_count"]
                ):

                    existing.update(
                        branch
                    )

                merged = True
                break

        if not merged:

            final_branches.append(
                branch
            )

    branch_count = len(
        final_branches
    )

    if not return_debug:

        return branch_count

    # --------------------------------------------------------
    # デバッグ用
    # --------------------------------------------------------

    debug_info = {

        "cn_image":
            cn_image,

        "branch_mask":
            branch_mask,

        "labels":
            labels,

        "accepted":
            final_branches,

        "branch_count":
            branch_count
    }

    return (
        branch_count,
        debug_info
    )


# ============================================================
# Endpoint
# ============================================================

def count_endpoints(
    skeleton
):

    """
    Cross Number = 1 の画素を endpoint候補とする。

    endpointについては、
    同一endpoint周辺の候補画素をまとめる。
    """

    skeleton = (
        skeleton > 0
    ).astype(
        np.uint8
    )

    endpoint_mask = np.zeros_like(
        skeleton,
        dtype=np.uint8
    )

    ys, xs = np.where(
        skeleton > 0
    )

    for y, x in zip(
        ys,
        xs
    ):

        cn = crossing_number_at(
            skeleton,
            int(y),
            int(x)
        )

        if cn == 1:

            endpoint_mask[y, x] = 1

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            endpoint_mask,
            connectivity=8
        )
    )

    endpoint_count = (
        num_labels - 1
    )

    return endpoint_count


# ============================================================
# Fractal Dimension
# ============================================================

def fractal_dimension(
    binary_image
):

    """
    Box-counting法によるFractal Dimension。
    """

    Z = (
        binary_image > 0
    )

    pixel_count = np.sum(
        Z
    )

    if pixel_count < 10:

        return (
            0.0,
            None,
            None
        )

    # --------------------------------------------------------
    # 正方形化
    # --------------------------------------------------------

    p = min(
        Z.shape[0],
        Z.shape[1]
    )

    n = 2 ** int(
        np.floor(
            np.log2(p)
        )
    )

    if n < 4:

        return (
            0.0,
            None,
            None
        )

    Z = Z[
        :n,
        :n
    ]

    # --------------------------------------------------------
    # Box size
    # --------------------------------------------------------

    sizes = 2 ** np.arange(
        int(np.log2(n)),
        1,
        -1
    )

    counts = []

    for size in sizes:

        S = np.add.reduceat(
            np.add.reduceat(
                Z,
                np.arange(
                    0,
                    Z.shape[0],
                    size
                ),
                axis=0
            ),
            np.arange(
                0,
                Z.shape[1],
                size
            ),
            axis=1
        )

        count = np.count_nonzero(
            S
        )

        counts.append(
            count
        )

    sizes = np.asarray(
        sizes,
        dtype=np.float64
    )

    counts = np.asarray(
        counts,
        dtype=np.float64
    )

    valid = (
        counts > 0
    )

    sizes = sizes[
        valid
    ]

    counts = counts[
        valid
    ]

    if len(counts) < 2:

        return (
            0.0,
            sizes,
            counts
        )

    # --------------------------------------------------------
    # 回帰
    # --------------------------------------------------------

    coeffs = np.polyfit(
        np.log(sizes),
        np.log(counts),
        1
    )

    fd = float(
        -coeffs[0]
    )

    return (
        fd,
        sizes,
        counts
    )


# ============================================================
# Fractal Dimension グラフ
# ============================================================

def save_fractal_plot(
    sizes,
    counts,
    fd,
    save_path,
    image_id
):

    if (
        sizes is None
        or counts is None
    ):
        return

    plt.figure(
        figsize=(7, 5)
    )

    x = np.log(
        sizes
    )

    y = np.log(
        counts
    )

    plt.scatter(
        x,
        y,
        s=45
    )

    if len(x) >= 2:

        coeffs = np.polyfit(
            x,
            y,
            1
        )

        x_fit = np.linspace(
            x.min(),
            x.max(),
            100
        )

        y_fit = (
            coeffs[0]
            * x_fit
            + coeffs[1]
        )

        plt.plot(
            x_fit,
            y_fit,
            linewidth=1
        )

    plt.xlabel(
        "log(Box Size)"
    )

    plt.ylabel(
        "log(Box Count)"
    )

    plt.title(
        f"Fractal Dimension\n"
        f"{image_id}   FD = {fd:.4f}"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()


# ============================================================
# Box-counting可視化
# ============================================================

def save_boxcount_visualization(
    skeleton,
    image_id,
    save_dir
):

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    binary = (
        skeleton > 0
    )

    p = min(
        binary.shape[0],
        binary.shape[1]
    )

    n = 2 ** int(
        np.floor(
            np.log2(p)
        )
    )

    if n < 32:
        return

    binary = binary[
        :n,
        :n
    ]

    box_sizes = [
        128,
        64,
        32,
        16
    ]

    for box_size in box_sizes:

        if box_size >= n:
            continue

        vis = cv2.cvtColor(
            binary.astype(
                np.uint8
            ) * 255,
            cv2.COLOR_GRAY2BGR
        )

        count = 0

        for y in range(
            0,
            n,
            box_size
        ):

            for x in range(
                0,
                n,
                box_size
            ):

                region = binary[
                    y:y + box_size,
                    x:x + box_size
                ]

                if np.any(region):

                    count += 1

                    cv2.rectangle(
                        vis,
                        (x, y),
                        (
                            min(
                                x + box_size,
                                n - 1
                            ),
                            min(
                                y + box_size,
                                n - 1
                            )
                        ),
                        (0, 0, 255),
                        1
                    )

        cv2.imwrite(
            os.path.join(
                save_dir,
                f"box_{box_size}"
                f"_count_{count}.png"
            ),
            vis
        )


# ============================================================
# Overlay
# ============================================================

def create_overlay(
    original,
    vessel,
    thickness=2
):

    vessel_bin = (
        vessel > 0
    )

    line = (
        vessel_bin.astype(
            np.uint8
        ) * 255
    )

    if thickness > 1:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                thickness,
                thickness
            )
        )

        line = cv2.dilate(
            line,
            kernel
        )

    overlay = original.copy()

    overlay[
        line > 0
    ] = (
        0,
        0,
        255
    )

    return overlay


# ============================================================
# 分岐点デバッグ画像
# ============================================================

def save_branch_debug(
    skeleton,
    save_path
):

    """
    分岐追跡の結果を可視化する。

    白:
        スケルトン

    赤:
        CN >= 3 の候補領域

    黄:
        最終的に「実際の分岐」と認定した中心

    緑:
        分岐中心からジャンプして追跡した枝

    青:
        ジャンプ開始点
    """

    skeleton = (
        skeleton > 0
    ).astype(
        np.uint8
    )

    branch_count, debug = (
        count_branch_points(
            skeleton,
            return_debug=True
        )
    )

    canvas = cv2.cvtColor(
        skeleton * 255,
        cv2.COLOR_GRAY2BGR
    )

    branch_mask = debug[
        "branch_mask"
    ]

    accepted = debug[
        "accepted"
    ]

    # --------------------------------------------------------
    # CN >= 3候補を赤
    # --------------------------------------------------------

    canvas[
        branch_mask > 0
    ] = (
        0,
        0,
        255
    )

    # --------------------------------------------------------
    # 最終分岐
    # --------------------------------------------------------

    for index, branch in enumerate(
        accepted,
        start=1
    ):

        cx = int(
            round(
                branch["x"]
            )
        )

        cy = int(
            round(
                branch["y"]
            )
        )

        # 分岐中心
        cv2.circle(
            canvas,
            (cx, cy),
            6,
            (0, 255, 255),
            -1
        )

        cv2.putText(
            canvas,
            str(index),
            (
                cx + 7,
                cy - 7
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # 追跡した枝
        # ----------------------------------------------------

        for branch_index, traced in enumerate(
            branch["branches"],
            start=1
        ):

            path = traced[
                "path"
            ]

            # 緑線
            for i in range(
                1,
                len(path)
            ):

                y1, x1 = path[
                    i - 1
                ]

                y2, x2 = path[
                    i
                ]

                cv2.line(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    1
                )

            # ジャンプ開始点
            sy, sx = traced[
                "start"
            ]

            cv2.circle(
                canvas,
                (sx, sy),
                3,
                (255, 0, 0),
                -1
            )

    cv2.putText(
        canvas,
        f"Branch Points = {branch_count}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imwrite(
        save_path,
        canvas
    )


# ============================================================
# Top10 個別保存
# ============================================================

def save_top20_evidence(
    df,
    original_map
):

    evidence_dir = os.path.join(
        TOP20_DIR,
        "individual"
    )

    os.makedirs(
        evidence_dir,
        exist_ok=True
    )

    print()
    print(
        "Top10の証拠画像を保存しています..."
    )

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Saving top10 evidence"
    ):

        rank = int(
            row["rank"]
        )

        image_id = str(
            row["image_id"]
        )

        vessel_path = str(
            row["vessel_path"]
        )

        original_path = original_map.get(
            image_id
        )

        rank_dir = os.path.join(
            evidence_dir,
            f"rank{rank:02d}_{image_id}"
        )

        os.makedirs(
            rank_dir,
            exist_ok=True
        )

        # ----------------------------------------------------
        # 血管
        # ----------------------------------------------------

        vessel = cv2.imread(
            vessel_path,
            cv2.IMREAD_GRAYSCALE
        )

        if vessel is None:
            continue

        cv2.imwrite(
            os.path.join(
                rank_dir,
                "vessel.png"
            ),
            vessel
        )

        # ----------------------------------------------------
        # 元画像
        # ----------------------------------------------------

        original = None

        if original_path is not None:

            original = cv2.imread(
                original_path,
                cv2.IMREAD_COLOR
            )

            if original is not None:

                cv2.imwrite(
                    os.path.join(
                        rank_dir,
                        "original.png"
                    ),
                    original
                )

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        if original is not None:

            if (
                original.shape[:2]
                != vessel.shape[:2]
            ):

                vessel_for_overlay = cv2.resize(
                    vessel,
                    (
                        original.shape[1],
                        original.shape[0]
                    ),
                    interpolation=cv2.INTER_NEAREST
                )

            else:

                vessel_for_overlay = vessel

            overlay = create_overlay(
                original,
                vessel_for_overlay,
                thickness=3
            )

            cv2.imwrite(
                os.path.join(
                    rank_dir,
                    "overlay.png"
                ),
                overlay
            )

        # ----------------------------------------------------
        # FD
        # ----------------------------------------------------

        fd, sizes, counts = (
            fractal_dimension(
                vessel
            )
        )

        save_fractal_plot(
            sizes,
            counts,
            fd,
            os.path.join(
                rank_dir,
                "fractal_dimension.png"
            ),
            image_id
        )

        # ----------------------------------------------------
        # Box count
        # ----------------------------------------------------

        save_boxcount_visualization(
            vessel,
            image_id,
            os.path.join(
                rank_dir,
                "boxcount"
            )
        )

        # ----------------------------------------------------
        # 分岐点デバッグ
        # ----------------------------------------------------

        save_branch_debug(
            vessel,
            os.path.join(
                rank_dir,
                "branch_points_debug.png"
            )
        )


# ============================================================
# Top10 Contact Sheet
# ============================================================

def save_top20_contact_sheet(
    df,
    original_map,
    save_path,
    image_type="original"
):

    cols = 5

    rows = int(
        np.ceil(
            len(df) / cols
        )
    )

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(
            20,
            4 * rows
        )
    )

    axes = np.asarray(
        axes
    ).reshape(-1)

    for i, (_, row) in enumerate(
        df.iterrows()
    ):

        ax = axes[i]

        image_id = str(
            row["image_id"]
        )

        vessel_path = str(
            row["vessel_path"]
        )

        original_path = original_map.get(
            image_id
        )

        image = None

        if image_type == "original":

            if original_path is not None:

                image = cv2.imread(
                    original_path
                )

                if image is not None:

                    image = cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2RGB
                    )

        elif image_type == "vessel":

            image = cv2.imread(
                vessel_path,
                cv2.IMREAD_GRAYSCALE
            )

        elif image_type == "overlay":

            if original_path is not None:

                original = cv2.imread(
                    original_path
                )

                vessel = cv2.imread(
                    vessel_path,
                    cv2.IMREAD_GRAYSCALE
                )

                if (
                    original is not None
                    and vessel is not None
                ):

                    if (
                        original.shape[:2]
                        != vessel.shape[:2]
                    ):

                        vessel = cv2.resize(
                            vessel,
                            (
                                original.shape[1],
                                original.shape[0]
                            ),
                            interpolation=cv2.INTER_NEAREST
                        )

                    image = create_overlay(
                        original,
                        vessel,
                        thickness=3
                    )

                    image = cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2RGB
                    )

        if image is not None:

            if image.ndim == 2:

                ax.imshow(
                    image,
                    cmap="gray"
                )

            else:

                ax.imshow(
                    image
                )

        else:

            ax.text(
                0.5,
                0.5,
                "Image not found",
                ha="center",
                va="center"
            )

        ax.axis("off")

        ax.set_title(
            f"Rank {int(row['rank'])}\n"
            f"FD = "
            f"{row['fractal_dimension']:.4f}\n"
            f"Branch = "
            f"{int(row['branch_points'])}\n"
            f"Score = "
            f"{row['complexity_score']:.2f}",
            fontsize=10
        )

    for i in range(
        len(df),
        len(axes)
    ):

        axes[i].axis(
            "off"
        )

    plt.suptitle(
        f"Top 10 Vessel Complexity - "
        f"{image_type}",
        fontsize=18
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96
        ]
    )

    plt.savefig(
        save_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# グラフ1
# ============================================================

def plot_top20_fd(
    df,
    save_path
):

    top = df[
        df["rank"] <= TOP_K
    ]

    plt.figure(
        figsize=(14, 7)
    )

    x = np.arange(
        len(top)
    )

    plt.bar(
        x,
        top["complexity_score"]
    )

    plt.xticks(
        x,
        [
            str(x)
            for x in top["rank"]
        ]
    )

    plt.xlabel(
        "Rank"
    )

    plt.ylabel(
        "Complexity Score"
    )

    plt.title(
        "Top 10 Vessel Complexity Score"
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()


# ============================================================
# グラフ2
# FD分布
# ============================================================

def plot_fd_distribution(
    df,
    save_path
):

    mean_fd = df[
        "fractal_dimension"
    ].mean()

    median_fd = df[
        "fractal_dimension"
    ].median()

    plt.figure(
        figsize=(12, 7)
    )

    plt.hist(
        df["fractal_dimension"],
        bins=40
    )

    plt.axvline(
        mean_fd,
        linestyle="--",
        label=f"Mean = {mean_fd:.4f}"
    )

    plt.axvline(
        median_fd,
        linestyle=":",
        label=f"Median = {median_fd:.4f}"
    )

    plt.xlabel(
        "Fractal Dimension"
    )

    plt.ylabel(
        "Number of Images"
    )

    plt.title(
        "Distribution of Vessel "
        "Fractal Dimension"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()


# ============================================================
# グラフ3
# ランク曲線
# ============================================================

def plot_rank_curve(
    df,
    save_path
):

    plt.figure(
        figsize=(14, 7)
    )

    plt.plot(
        df["rank"],
        df["complexity_score"],
        linewidth=1
    )

    top = df[
        df["rank"] <= TOP_K
    ]

    plt.scatter(
        top["rank"],
        top["complexity_score"],
        s=30,
        label="Top 10"
    )

    plt.xlabel(
        "Rank"
    )

    plt.ylabel(
        "Complexity Score"
    )

    plt.title(
        "Vessel Complexity Ranking"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()


# ============================================================
# グラフ4
# FD vs Branch
# ============================================================

def plot_fd_vs_branch(
    df,
    save_path
):

    plt.figure(
        figsize=(9, 7)
    )

    plt.scatter(
        df["fd_score"],
        df["branch_score"],
        s=10,
        alpha=0.5
    )

    top = df[
        df["rank"] <= TOP_K
    ]

    plt.scatter(
        top["fd_score"],
        top["branch_score"],
        s=45,
        label="Top 10"
    )

    plt.xlabel(
        "FD Score (0-100)"
    )

    plt.ylabel(
        "Branch Score (0-100)"
    )

    plt.title(
        "FD Score vs Branch Score"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()


# ============================================================
# グラフ5
# FD vs Skeleton Length
# ============================================================

def plot_fd_vs_length(
    df,
    save_path
):

    plt.figure(
        figsize=(9, 7)
    )

    plt.scatter(
        df["fractal_dimension"],
        df["skeleton_length"],
        s=10,
        alpha=0.5
    )

    top = df[
        df["rank"] <= TOP_K
    ]

    plt.scatter(
        top["fractal_dimension"],
        top["skeleton_length"],
        s=45,
        label="Top 10"
    )

    plt.xlabel(
        "Fractal Dimension"
    )

    plt.ylabel(
        "Skeleton Length"
    )

    plt.title(
        "Fractal Dimension vs "
        "Skeleton Length"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()


# ============================================================
# グラフ6
# Top10の位置
# ============================================================

def plot_top20_boxplot(
    df,
    save_path
):

    top = df[
        df["rank"] <= TOP_K
    ]

    plt.figure(
        figsize=(7, 7)
    )

    plt.boxplot(
        df["fractal_dimension"]
    )

    x = np.ones(
        len(top)
    )

    plt.scatter(
        x,
        top["fractal_dimension"],
        s=40,
        label="Top 10"
    )

    plt.ylabel(
        "Fractal Dimension"
    )

    plt.title(
        "Top 10 Position in "
        "Overall FD Distribution"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("血管複雑性ランキング")
    print("=" * 70)

    print()
    print(
        "分岐検出方式:"
    )

    print(
        "  CN >= 3 で分岐候補を検出"
    )

    print(
        "  ↓"
    )

    print(
        "  候補から外側へジャンプ"
    )

    print(
        "  ↓"
    )

    print(
        "  血管を追跡"
    )

    print(
        "  ↓"
    )

    print(
        "  3本以上の独立枝を確認した場合のみ分岐"
    )

    print()

    # --------------------------------------------------------
    # 入力確認
    # --------------------------------------------------------

    if not os.path.isdir(
        SOURCE_IMAGE_DIR
    ):

        raise FileNotFoundError(
            f"元画像フォルダがありません: "
            f"{SOURCE_IMAGE_DIR}"
        )

    if not os.path.isdir(
        VESSEL_DIR
    ):

        raise FileNotFoundError(
            f"血管画像フォルダがありません: "
            f"{VESSEL_DIR}"
        )

    # --------------------------------------------------------
    # 出力フォルダ
    # --------------------------------------------------------

    clear_dir(
        TOP20_DIR
    )

    clear_dir(
        GRAPH_DIR
    )

    # --------------------------------------------------------
    # 元画像
    # --------------------------------------------------------

    original_map = (
        build_original_image_map()
    )

    print(
        f"元画像数: "
        f"{len(original_map)}"
    )

    # --------------------------------------------------------
    # 血管画像
    # --------------------------------------------------------

    vessel_files = sorted(
        f
        for f in os.listdir(
            VESSEL_DIR
        )
        if f.lower().endswith(
            ".png"
        )
        and f.endswith(
            "_vessel.png"
        )
    )

    if not vessel_files:

        raise FileNotFoundError(
            "vessel_extracted 内に "
            "*_vessel.png がありません。"
        )

    print(
        f"血管画像数: "
        f"{len(vessel_files)}"
    )

    print(
        "複雑さスコア: FD 70% + 分岐数 30%"
    )

    print(
        "分岐点: "
        "Cross Number + "
        "ジャンプ + 血管追跡"
    )

    print()

    # --------------------------------------------------------
    # 解析
    # --------------------------------------------------------

    results = []

    for vessel_file in tqdm(
        vessel_files,
        desc="Complexity analysis"
    ):

        vessel_path = os.path.join(
            VESSEL_DIR,
            vessel_file
        )

        vessel = cv2.imread(
            vessel_path,
            cv2.IMREAD_GRAYSCALE
        )

        if vessel is None:

            print(
                f"[SKIP] 読み込み失敗: "
                f"{vessel_file}"
            )

            continue

        image_id = (
            get_image_id_from_vessel(
                vessel_file
            )
        )

        # ----------------------------------------------------
        # FD
        # ----------------------------------------------------

        fd, sizes, counts = (
            fractal_dimension(
                vessel
            )
        )

        # ----------------------------------------------------
        # 分岐
        # ----------------------------------------------------

        branch_points = (
            count_branch_points(
                vessel
            )
        )

        # ----------------------------------------------------
        # Endpoint
        # ----------------------------------------------------

        end_points = (
            count_endpoints(
                vessel
            )
        )

        # ----------------------------------------------------
        # Skeleton length
        # ----------------------------------------------------

        skeleton_length = int(
            np.sum(
                vessel > 0
            )
        )

        results.append({

            "image_id":
                image_id,

            "original_file":
                os.path.basename(
                    original_map.get(
                        image_id,
                        ""
                    )
                ),

            "vessel_file":
                vessel_file,

            "fractal_dimension":
                fd,

            "branch_points":
                branch_points,

            "end_points":
                end_points,

            "skeleton_length":
                skeleton_length,

            "vessel_pixel_count":
                skeleton_length,

            "vessel_path":
                vessel_path
        })

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        results
    )

    if len(df) == 0:

        raise RuntimeError(
            "有効な血管画像を1枚も処理できませんでした。"
        )

    # --------------------------------------------------------
    # 有効画像
    # --------------------------------------------------------

    df["valid"] = (
        (df["fractal_dimension"] > 0)
        &
        (df["skeleton_length"] > 0)
    )

    invalid_count = int(
        (~df["valid"]).sum()
    )

    if invalid_count > 0:

        print()
        print(
            "[WARNING] "
            f"有効でない画像: "
            f"{invalid_count} 枚"
        )

    # --------------------------------------------------------
    # 複雑さスコア
    #
    # FD 70% + 分岐数 30%
    #
    # FDと分岐数をそれぞれ全画像内で0～100に正規化してから
    # 加重平均する。
    #
    # skeleton_length はランキングに一切使用しない。
    # --------------------------------------------------------

    valid_mask = (
        (df["fractal_dimension"] > 0)
        &
        (df["skeleton_length"] > 0)
    )

    df["valid"] = valid_mask

    invalid_count = int(
        (~df["valid"]).sum()
    )

    if invalid_count > 0:

        print()
        print(
            "[WARNING] "
            f"有効でない画像: "
            f"{invalid_count} 枚"
        )

    # --------------------------------------------------------
    # 有効画像についてFDと分岐数を0～100に正規化
    # --------------------------------------------------------

    valid_df = df[
        df["valid"]
    ].copy()

    if len(valid_df) == 0:

        raise RuntimeError(
            "有効な画像が1枚もありません。"
        )

    fd_min = valid_df[
        "fractal_dimension"
    ].min()

    fd_max = valid_df[
        "fractal_dimension"
    ].max()

    branch_min = valid_df[
        "branch_points"
    ].min()

    branch_max = valid_df[
        "branch_points"
    ].max()

    if fd_max > fd_min:

        df["fd_score"] = (
            (df["fractal_dimension"] - fd_min)
            /
            (fd_max - fd_min)
            * 100.0
        )

    else:

        df["fd_score"] = 0.0

    if branch_max > branch_min:

        df["branch_score"] = (
            (df["branch_points"] - branch_min)
            /
            (branch_max - branch_min)
            * 100.0
        )

    else:

        df["branch_score"] = 0.0

    # 無効画像はランキング対象外
    df.loc[
        ~df["valid"],
        [
            "fd_score",
            "branch_score"
        ]
    ] = 0.0

    # --------------------------------------------------------
    # 最終複雑さスコア
    # --------------------------------------------------------

    df["complexity_score"] = (
        0.70 * df["fd_score"]
        +
        0.30 * df["branch_score"]
    )

    # --------------------------------------------------------
    # 複雑さスコアの高い順にランキング
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "valid",
            "complexity_score",
            "fractal_dimension",
            "branch_points"
        ],
        ascending=[
            False,
            False,
            False,
            False
        ]
    ).reset_index(
        drop=True
    )

    df["rank"] = np.arange(
        1,
        len(df) + 1
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_columns = [

        "rank",

        "image_id",

        "original_file",

        "vessel_file",

        "fractal_dimension",

        "fd_score",

        "branch_points",

        "branch_score",

        "complexity_score",

        "end_points",

        "skeleton_length",

        "vessel_pixel_count",

        "valid"
    ]

    df[
        csv_columns
    ].to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Top10
    # --------------------------------------------------------

    top20 = df.head(
        TOP_K
    ).copy()

    top20.to_csv(
        os.path.join(
            TOP20_DIR,
            "top20.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Top10証拠
    # --------------------------------------------------------

    save_top20_evidence(
        top20,
        original_map
    )

    # --------------------------------------------------------
    # Contact Sheet
    # --------------------------------------------------------

    print()
    print(
        "Top10の一覧画像を作成しています..."
    )

    save_top20_contact_sheet(
        top20,
        original_map,
        os.path.join(
            TOP20_DIR,
            "01_top10_original.png"
        ),
        image_type="original"
    )

    save_top20_contact_sheet(
        top20,
        original_map,
        os.path.join(
            TOP20_DIR,
            "02_top10_vessel.png"
        ),
        image_type="vessel"
    )

    save_top20_contact_sheet(
        top20,
        original_map,
        os.path.join(
            TOP20_DIR,
            "03_top10_overlay.png"
        ),
        image_type="overlay"
    )

    # --------------------------------------------------------
    # グラフ
    # --------------------------------------------------------

    print()
    print(
        "根拠となるグラフを作成しています..."
    )

    plot_top20_fd(
        df,
        os.path.join(
            GRAPH_DIR,
            "01_top20_fractal_dimension.png"
        )
    )

    plot_fd_distribution(
        df,
        os.path.join(
            GRAPH_DIR,
            "02_fractal_dimension_distribution.png"
        )
    )

    plot_rank_curve(
        df,
        os.path.join(
            GRAPH_DIR,
            "03_fractal_dimension_rank_curve.png"
        )
    )

    plot_fd_vs_branch(
        df,
        os.path.join(
            GRAPH_DIR,
            "04_fd_vs_branch_points.png"
        )
    )

    plot_top20_boxplot(
        df,
        os.path.join(
            GRAPH_DIR,
            "06_top20_position_in_distribution.png"
        )
    )

    # --------------------------------------------------------
    # 結果
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("完了")
    print("=" * 70)

    print()
    print(
        f"全結果CSV: "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Top10: "
        f"{TOP20_DIR}/"
    )

    print(
        f"グラフ: "
        f"{GRAPH_DIR}/"
    )

    print()
    print(
        "Top 10:"
    )

    print(
        top20[
            [
                "rank",
                "image_id",
                "fractal_dimension",
                "fd_score",
                "branch_points",
                "branch_score",
                "complexity_score"
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "分岐数の計算方法:"
    )

    print(
        "  1. 8近傍のCrossing Numberを計算"
    )

    print(
        "  2. CN >= 3 を分岐候補とする"
    )

    print(
        "  3. 候補領域から外側へジャンプ"
    )

    print(
        "  4. ジャンプ地点から血管を追跡"
    )

    print(
        "  5. 3本以上の独立した枝を確認"
    )

    print(
        "  6. 3本以上なら1つの分岐として認定"
    )

    print(
        "  7. 近接する同一分岐候補を重複除去"
    )

    print()
    print(
        "作成されたグラフ:"
    )

    for name in sorted(
        os.listdir(
            GRAPH_DIR
        )
    ):

        print(
            f"  "
            f"{GRAPH_DIR}/{name}"
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()
