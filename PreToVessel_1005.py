import os
import random
import shutil
import re

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from skimage.filters import frangi
    from skimage.morphology import skeletonize
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals
except ImportError as exc:
    raise ImportError(
        "scikit-image が必要です。次を実行してください: pip install scikit-image"
    ) from exc



# ============================================================
# 基本設定
# ============================================================

# 入力画像フォルダ
# ユーザー指定に合わせて healthy_images を使用
SOURCE_IMAGE_DIR = "healthy_images"

SUPPORTED_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
)

# ------------------------------------------------------------
# 出力
# ------------------------------------------------------------

# 通常時は血管抽出画像だけを保存する。
VESSEL_OUTPUT_DIR = "vessel_extracted"

# デバッグ用は、先頭20件について「全工程 + 血管抽出結果 + 元眼底画像」
# を1枚にまとめた画像だけを保存する。
DEBUG_MONTAGE_DIR = "vessel_debug_20"

# ------------------------------------------------------------
# 色統計・外れ値
# ------------------------------------------------------------

# カラー判定の根拠として、統計CSV・箱ひげ図・散布図・
# 外れ値画像は保存して残す。
COLOR_STATISTICS_CSV = "color_statistics.csv"
COLOR_OUTLIERS_CSV = "color_outliers.csv"
COLOR_VISUALIZATION_DIR = "color_visualization"
COLOR_BOXPLOT_DIR = os.path.join(
    COLOR_VISUALIZATION_DIR, "boxplots"
)
COLOR_OUTLIER_PREVIEW_DIR = os.path.join(
    COLOR_VISUALIZATION_DIR, "outlier_previews"
)

COLOR_IQR_MULTIPLIER = 1.5
COLOR_OUTLIER_FEATURE_COUNT = 2

# ------------------------------------------------------------
# Debug
# ------------------------------------------------------------

DEBUG_SAMPLE_COUNT = 20
RANDOM_SEED = 42

# ============================================================
# 2本目の血管抽出ロジックの設定
# ============================================================

FUNDUS_THRESHOLD = 10
FUNDUS_CLOSE_SIZE = 21
FUNDUS_ERODE_SIZE = 7
EFFECTIVE_MASK_ERODE_SIZE = 9
BORDER_SUPPRESSION_WIDTH = 18

DARK_MEAN_THRESHOLD = 92.0
NOISE_STD_THRESHOLD = 10.0
ILLUMINATION_SIGMA = 35.0
LOCAL_BACKGROUND_SIGMA = 9.0

CLAHE_CLIP_NORMAL = 1.25
CLAHE_CLIP_DIFFICULT = 0.85
CLAHE_TILE_GRID = (12, 12)

FRANGI_SIGMAS = (0.8, 1.1, 1.5, 2.0, 2.8, 3.8)
FRANGI_ALPHA = 0.5
FRANGI_BETA = 0.5
FRANGI_GAMMA = None

ORIENTED_LINE_LENGTHS = (9, 15, 23)
ORIENTED_LINE_ANGLE_STEP = 15
BLACKHAT_KERNEL_SIZES = (11, 17, 25, 31)
DARK_LINE_SIGMA = 9.0
COHERENCE_SIGMA_NORMAL = 5.0
COHERENCE_SIGMA_DIFFICULT = 9.0

LOCAL_RELATIVE_SIGMA = 23.0
LOCAL_RELATIVE_MEAN_FLOOR = 3.0

STRONG_FRANGI_PERCENTILE = 88.0
WEAK_FRANGI_PERCENTILE = 50.0
BLACKHAT_PERCENTILE = 68.0
DARK_PERCENTILE = 58.0

COHERENCE_STRONG_MIN = 0.34
COHERENCE_WEAK_MIN = 0.28
LOCAL_RELATIVE_MIN = 1.65

RING_PENALTY_ENABLED = True
RING_CONDITION_MIN = 0.40
RING_SCORE_THRESHOLD = 0.82

HYSTERESIS_MAX_ITERATIONS = 512
MIN_COMPONENT_AREA = 8
MIN_COMPONENT_LENGTH = 7
MIN_COMPONENT_MEAN_FRANGI = 10.0
CENTERLINE_CLOSE_SIZE = 3

DEBUG_TILE_SIZE = (280, 280)
DEBUG_COLUMNS = 4
DEBUG_TITLE_HEIGHT = 24

# 最終overlayの表示線幅
OUTPUT_LINE_THICKNESS = 3
DEBUG_OVERLAY_LINE_THICKNESS = 5
OVERLAY_ALPHA = 0.88

# 通常出力は血管抽出画像そのもの。
OUTPUT_IMAGE_MODE = "vessel_only"

CLEAR_OUTPUTS_AT_START = True



# ============================================================
# 1本目: 共通ユーティリティ・前処理・色解析
# ============================================================


def odd_size(value):
    """OpenCVのカーネル用に、指定値以上の奇数へ丸める。"""
    value = max(1, int(round(value)))
    return value if value % 2 == 1 else value + 1


def safe_erode(mask, kernel, iterations=1):

    kh, kw = kernel.shape[:2]
    pad = max(kh, kw)

    padded = cv2.copyMakeBorder(
        mask,
        pad,
        pad,
        pad,
        pad,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    eroded = cv2.erode(
        padded,
        kernel,
        iterations=iterations,
    )

    return eroded[
        pad:pad + mask.shape[0],
        pad:pad + mask.shape[1],
    ]




def safe_dilate(mask, kernel, iterations=1):

    kh, kw = kernel.shape[:2]
    pad = max(kh, kw)

    padded = cv2.copyMakeBorder(
        mask,
        pad,
        pad,
        pad,
        pad,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    dilated = cv2.dilate(
        padded,
        kernel,
        iterations=iterations,
    )

    return dilated[
        pad:pad + mask.shape[0],
        pad:pad + mask.shape[1],
    ]




def safe_close(mask, kernel):

    kh, kw = kernel.shape[:2]
    pad = max(kh, kw)

    padded = cv2.copyMakeBorder(
        mask,
        pad,
        pad,
        pad,
        pad,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    closed = cv2.morphologyEx(
        padded,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return closed[
        pad:pad + mask.shape[0],
        pad:pad + mask.shape[1],
    ]




def normalize_01(image, mask=None):

    image = image.astype(np.float32)

    if mask is not None:
        values = image[mask > 0]
    else:
        values = image.ravel()

    if values.size == 0:
        return np.zeros_like(
            image,
            dtype=np.float32
        )

    lo = np.percentile(
        values,
        1.0
    )

    hi = np.percentile(
        values,
        99.0
    )

    if hi <= lo + 1e-6:
        return np.zeros_like(
            image,
            dtype=np.float32
        )

    result = (
        image - lo
    ) / (
        hi - lo
    )

    result = np.clip(
        result,
        0.0,
        1.0
    )

    if mask is not None:

        result[
            mask == 0
        ] = 0.0

    return result.astype(
        np.float32
    )




def to_uint8(image):

    image = np.clip(
        image,
        0.0,
        1.0
    )

    return (
        image * 255.0
    ).astype(
        np.uint8
    )




def mask_image(image, mask):
    """画像をマスク領域だけに制限する。"""
    mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)

    if image.ndim == 2:
        return cv2.bitwise_and(
            image,
            image,
            mask=mask_u8,
        )

    return cv2.bitwise_and(
        image,
        image,
        mask=mask_u8,
    )



def threshold_percentile(
    response,
    mask,
    percentile
):

    values = response[
        mask > 0
    ]

    if values.size == 0:

        return np.zeros_like(
            mask
        )

    threshold = np.percentile(
        values,
        percentile
    )

    binary = (
        (response >= threshold)
        & (mask > 0)
    )

    return (
        binary.astype(
            np.uint8
        ) * 255
    )




def remove_small_components(
    binary,
    min_area
):

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    output = np.zeros_like(
        binary
    )

    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= min_area:

            output[
                labels == label
            ] = 255

    return output




class FundusPreprocessor:

    def __init__(
        self,
        image_dir=SOURCE_IMAGE_DIR,
        output_color_dir=None,
        output_green_dir=None,
        output_mask_dir=None,
        debug_dir=None,
        debug_sample_count=DEBUG_SAMPLE_COUNT,
        random_seed=RANDOM_SEED,
    ):

        self.image_dir = image_dir

        self.output_color_dir = (
            output_color_dir
        )

        self.output_green_dir = (
            output_green_dir
        )

        self.output_mask_dir = (
            output_mask_dir
        )

        self.debug_dir = debug_dir

        self.debug_sample_count = (
            debug_sample_count
        )

        self.random_seed = (
            random_seed
        )

        # 出力保存はmain側で必要な画像だけ行う。
        # 中間画像用フォルダは作成しない。

    # --------------------------------------------------------
    # 眼底マスク
    # --------------------------------------------------------

    def create_fundus_mask(
        self,
        img
    ):
        """眼底画像の有効領域だけを残すマスクを作る。

        単純な固定閾値だけだと、JPEG圧縮や黒背景の持ち上がりによって
        画像全体が前景と判定され、矩形の外側まで血管抽出対象になることがある。
        そこで、Otsuによる候補領域→最大連結成分→境界接触時の楕円制約→
        最終的な軽い収縮、という順で眼底外周を明示的に除外する。
        """

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        # わずかな圧縮ノイズを抑えてから眼底と背景を分離する。
        gray_smooth = cv2.GaussianBlur(
            gray,
            (0, 0),
            sigmaX=1.5,
            sigmaY=1.5,
        )

        otsu_value, _ = cv2.threshold(
            gray_smooth,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        # 極端に暗い/明るい画像でも固定閾値から大きく外れないようにする。
        threshold_value = float(
            max(
                FUNDUS_THRESHOLD,
                min(float(otsu_value), 80.0),
            )
        )

        _, mask = cv2.threshold(
            gray_smooth,
            threshold_value,
            255,
            cv2.THRESH_BINARY
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                odd_size(FUNDUS_CLOSE_SIZE),
                odd_size(FUNDUS_CLOSE_SIZE),
            ),
        )

        mask = safe_close(
            mask,
            kernel
        )

        # 小さな背景ノイズを除去して最大の眼底領域だけ残す。
        mask = keep_largest_component(mask)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        clean_mask = np.zeros_like(mask)

        if len(contours) == 0:
            return clean_mask

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        cv2.drawContours(
            clean_mask,
            [largest_contour],
            -1,
            255,
            thickness=-1
        )

        # ----------------------------------------------------
        # 重要: 候補領域が画像端まで広がった場合の補正
        # ----------------------------------------------------
        #
        # ここが従来処理の弱点。背景が完全な0でない画像では、
        # threshold後の最大輪郭が画像全体の矩形になり得る。
        # その場合は輪郭から眼底の楕円形を推定し、矩形部分を切り落とす。
        # ----------------------------------------------------
        h, w = gray.shape[:2]
        image_area = float(max(h * w, 1))
        contour_area = float(cv2.contourArea(largest_contour))
        area_ratio = contour_area / image_area

        touches_border = bool(
            np.any(clean_mask[0, :] > 0)
            or np.any(clean_mask[-1, :] > 0)
            or np.any(clean_mask[:, 0] > 0)
            or np.any(clean_mask[:, -1] > 0)
        )

        if touches_border or area_ratio > 0.97:
            ellipse_mask = np.zeros_like(clean_mask)

            if len(largest_contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(largest_contour)
                    (cx, cy), (axis_a, axis_b), angle = ellipse

                    # 楕円を輪郭いっぱいにしないことで、外周の縁を避ける。
                    axis_a = max(float(axis_a) * 0.97, 10.0)
                    axis_b = max(float(axis_b) * 0.97, 10.0)

                    cv2.ellipse(
                        ellipse_mask,
                        (
                            int(round(cx)),
                            int(round(cy)),
                        ),
                        (
                            max(1, int(round(axis_a / 2.0))),
                            max(1, int(round(axis_b / 2.0))),
                        ),
                        float(angle),
                        0,
                        360,
                        255,
                        -1,
                    )
                except cv2.error:
                    ellipse_mask = np.zeros_like(clean_mask)

            # fitEllipseが使えない場合は画像中央の楕円を安全側に採用。
            if np.count_nonzero(ellipse_mask) == 0:
                center = (w // 2, h // 2)
                axes = (
                    max(1, int(round(w * 0.475))),
                    max(1, int(round(h * 0.475))),
                )
                cv2.ellipse(
                    ellipse_mask,
                    center,
                    axes,
                    0,
                    0,
                    360,
                    255,
                    -1,
                )

            clean_mask = cv2.bitwise_and(
                clean_mask,
                ellipse_mask,
            )

        # 最終的に外周へ少し余裕を持たせる。
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                odd_size(FUNDUS_ERODE_SIZE),
                odd_size(FUNDUS_ERODE_SIZE),
            ),
        )

        clean_mask = safe_erode(
            clean_mask,
            erode_kernel,
            iterations=1
        )

        # 万一まだ画像端に接していたら、画像端から数画素を強制除外する。
        # 通常の眼底画像ではこの処理はほぼ発生しない。
        if (
            np.any(clean_mask[0, :] > 0)
            or np.any(clean_mask[-1, :] > 0)
            or np.any(clean_mask[:, 0] > 0)
            or np.any(clean_mask[:, -1] > 0)
        ):
            border = max(
                1,
                int(round(min(h, w) * 0.01))
            )
            clean_mask[:border, :] = 0
            clean_mask[-border:, :] = 0
            clean_mask[:, :border] = 0
            clean_mask[:, -border:] = 0

        return clean_mask

    # --------------------------------------------------------
    # マスク境界接触
    # --------------------------------------------------------

    def mask_touches_border(
        self,
        mask
    ):

        if mask.size == 0:

            return False

        return bool(
            np.any(mask[0, :] > 0)
            or np.any(mask[-1, :] > 0)
            or np.any(mask[:, 0] > 0)
            or np.any(mask[:, -1] > 0)
        )

    # --------------------------------------------------------
    # 基本統計
    # --------------------------------------------------------

    def calculate_metrics(
        self,
        green,
        mask
    ):

        pixels = green[
            mask > 0
        ]

        if pixels.size == 0:

            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "dark_ratio": 1.0,
            }

        return {
            "mean": float(
                np.mean(pixels)
            ),

            "std": float(
                np.std(pixels)
            ),

            "min": float(
                np.min(pixels)
            ),

            "max": float(
                np.max(pixels)
            ),

            "dark_ratio": float(
                np.mean(
                    pixels < 30
                )
            ),
        }

    # --------------------------------------------------------
    # ノイズ推定
    # --------------------------------------------------------

    def estimate_noise_std(
        self,
        gray,
        mask
    ):

        if np.count_nonzero(
            mask
        ) == 0:

            return 0.0

        laplacian = cv2.Laplacian(
            gray,
            cv2.CV_64F,
            ksize=3
        )

        values = laplacian[
            mask > 0
        ]

        if values.size == 0:

            return 0.0

        return float(
            np.std(values)
        )

    # --------------------------------------------------------
    # 暗い/ノイジー画像用事前denoise
    # --------------------------------------------------------

    def adaptive_pre_denoise(
        self,
        green,
        mask,
        before_metrics,
        noise_std,
    ):

        is_dark = (
            before_metrics["mean"]
            < 90.0
        )

        is_noisy = (
            noise_std
            > 6.0
        )

        if not (
            is_dark
            or is_noisy
        ):

            return (
                green.copy(),
                0.0
            )

        strength = 0.0

        if is_dark:

            strength += max(
                0.0,
                (
                    90.0
                    - before_metrics["mean"]
                ) / 90.0
            )

        if is_noisy:

            strength += min(
                noise_std / 20.0,
                1.0
            )

        strength = float(
            np.clip(
                strength,
                0.0,
                1.0
            )
        )

        h = (
            4.0
            + strength * 8.0
        )

        denoised = (
            cv2.fastNlMeansDenoising(
                green,
                None,
                h=h,
                templateWindowSize=7,
                searchWindowSize=21,
            )
        )

        denoised = cv2.bitwise_and(
            denoised,
            denoised,
            mask=mask
        )

        return (
            denoised,
            h
        )

    # --------------------------------------------------------
    # illumination correction
    # --------------------------------------------------------

    def illumination_correction(
        self,
        green,
        fundus_mask,
        dither_amplitude=0.75,
    ):

        green_masked = cv2.bitwise_and(
            green,
            green,
            mask=fundus_mask
        )

        green_float = (
            green_masked.astype(
                np.float32
            )
        )

        background = cv2.GaussianBlur(
            green_float,
            (0, 0),
            sigmaX=35,
            sigmaY=35
        )

        corrected = (
            green_float
            / (
                background
                + 1.0
            )
        ) * 128.0

        if dither_amplitude > 0:

            dither = np.random.uniform(
                -dither_amplitude,
                dither_amplitude,
                size=corrected.shape
            ).astype(
                np.float32
            )

            corrected += dither

        corrected = np.clip(
            corrected,
            0,
            255
        ).astype(
            np.uint8
        )

        corrected = cv2.bitwise_and(
            corrected,
            corrected,
            mask=fundus_mask
        )

        return (
            corrected,
            background.astype(
                np.uint8
            )
        )

    # --------------------------------------------------------
    # 輝度補正
    # --------------------------------------------------------

    def brightness_adjustment(
        self,
        green,
        fundus_mask,
        target_mean=120,
        max_scale=2.8,
    ):

        adjusted = green.copy()

        pixels = green[
            fundus_mask > 0
        ]

        if pixels.size == 0:

            return (
                adjusted,
                1.0
            )

        current_mean = (
            np.mean(pixels)
        )

        if current_mean <= 1:

            return (
                adjusted,
                1.0
            )

        scale = (
            target_mean
            / current_mean
        )

        scale = float(
            min(
                scale,
                max_scale
            )
        )

        adjusted = (
            green.astype(
                np.float32
            )
            * scale
        )

        adjusted = np.clip(
            adjusted,
            0,
            255
        ).astype(
            np.uint8
        )

        adjusted = cv2.bitwise_and(
            adjusted,
            adjusted,
            mask=fundus_mask
        )

        return (
            adjusted,
            scale
        )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    def apply_clahe(
        self,
        green,
        fundus_mask,
        clip_limit=1.5
    ):

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(8, 8)
        )

        result = clahe.apply(
            green
        )

        result = cv2.bitwise_and(
            result,
            result,
            mask=fundus_mask
        )

        return result

    # --------------------------------------------------------
    # 最終denoise
    # --------------------------------------------------------

    def gentle_denoise(
        self,
        green,
        fundus_mask,
        strength=1.0
    ):

        sigma = (
            15.0
            + strength * 10.0
        )

        result = cv2.bilateralFilter(
            green,
            d=5,
            sigmaColor=sigma,
            sigmaSpace=sigma
        )

        result = cv2.bitwise_and(
            result,
            result,
            mask=fundus_mask
        )

        return result

    # --------------------------------------------------------
    # カラー画像
    # --------------------------------------------------------

    def build_color_preprocessed_image(
        self,
        original_bgr,
        green_preprocessed,
        fundus_mask
    ):

        color = original_bgr.copy()

        color[:, :, 1] = (
            green_preprocessed
        )

        color = cv2.bitwise_and(
            color,
            color,
            mask=fundus_mask
        )

        return color

    # --------------------------------------------------------
    # 1枚処理
    # --------------------------------------------------------

    def preprocess_one_image(
        self,
        img
    ):

        steps = {}

        steps[
            "01_original"
        ] = img.copy()

        fundus_mask = (
            self.create_fundus_mask(
                img
            )
        )

        steps[
            "02_fundus_mask"
        ] = fundus_mask.copy()

        touches_border = (
            self.mask_touches_border(
                fundus_mask
            )
        )

        green_raw = img[:, :, 1]

        steps[
            "03_green_raw"
        ] = green_raw.copy()

        before_metrics = (
            self.calculate_metrics(
                green_raw,
                fundus_mask
            )
        )

        noise_std = (
            self.estimate_noise_std(
                green_raw,
                fundus_mask
            )
        )

        (
            green_pre_denoised,
            pre_denoise_h
        ) = self.adaptive_pre_denoise(
            green_raw,
            fundus_mask,
            before_metrics,
            noise_std
        )

        steps[
            "03b_green_pre_denoised"
        ] = green_pre_denoised.copy()

        (
            green_corrected,
            background
        ) = self.illumination_correction(
            green_pre_denoised,
            fundus_mask
        )

        steps[
            "04_green_illumination_corrected"
        ] = green_corrected.copy()

        (
            green_brightness,
            brightness_scale
        ) = self.brightness_adjustment(
            green_corrected,
            fundus_mask,
            target_mean=120,
            max_scale=2.8
        )

        steps[
            "05_green_brightness_adjusted"
        ] = green_brightness.copy()

        is_dark = (
            before_metrics["mean"]
            < 90.0
        )

        is_noisy = (
            noise_std
            > 6.0
        )

        clip_limit = (
            1.0
            if (
                is_dark
                or is_noisy
            )
            else 1.5
        )

        green_clahe = (
            self.apply_clahe(
                green_brightness,
                fundus_mask,
                clip_limit=clip_limit
            )
        )

        steps[
            "06_green_clahe"
        ] = green_clahe.copy()

        denoise_strength = (
            1.0
            if (
                is_dark
                or is_noisy
            )
            else 0.0
        )

        green_preprocessed = (
            self.gentle_denoise(
                green_clahe,
                fundus_mask,
                strength=denoise_strength
            )
        )

        steps[
            "07_green_preprocessed"
        ] = green_preprocessed.copy()

        color_preprocessed = (
            self.build_color_preprocessed_image(
                img,
                green_preprocessed,
                fundus_mask
            )
        )

        steps[
            "08_color_preprocessed"
        ] = color_preprocessed.copy()

        after_metrics = (
            self.calculate_metrics(
                green_preprocessed,
                fundus_mask
            )
        )

        diagnostics = {

            "noise_std":
                noise_std,

            "pre_denoise_h":
                pre_denoise_h,

            "brightness_scale":
                brightness_scale,

            "clahe_clip_limit":
                clip_limit,

            "is_dark":
                is_dark,

            "is_noisy":
                is_noisy,

            "touches_border":
                touches_border,
        }

        return (
            green_preprocessed,
            color_preprocessed,
            fundus_mask,
            steps,
            before_metrics,
            after_metrics,
            diagnostics,
        )

    # --------------------------------------------------------
    # デバッグ保存
    # --------------------------------------------------------

    def save_debug_steps(
        self,
        steps,
        save_dir
    ):

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        for name, image in steps.items():

            path = os.path.join(
                save_dir,
                name + ".png"
            )

            cv2.imwrite(
                path,
                image
            )




class ColorAnalyzer:

    def __init__(
        self,
        iqr_multiplier=1.5,
        outlier_feature_count=2
    ):

        self.iqr_multiplier = (
            iqr_multiplier
        )

        self.outlier_feature_count = (
            outlier_feature_count
        )

        # ----------------------------------------------------
        # IQRによる外れ値判定に使用する特徴量
        # ----------------------------------------------------

        self.features = [

            "mean_R",
            "mean_G",
            "mean_B",

            "mean_L",
            "mean_a",
            "mean_b",

            "mean_S",
            "mean_V",

        ]

    # --------------------------------------------------------
    # 色統計計算
    # --------------------------------------------------------

    def calculate_color_statistics(
        self,
        image,
        mask
    ):

        pixels_mask = (
            mask > 0
        )

        if np.count_nonzero(
            pixels_mask
        ) == 0:

            return None

        # ----------------------------------------------------
        # BGR
        # ----------------------------------------------------

        b = image[:, :, 0][
            pixels_mask
        ]

        g = image[:, :, 1][
            pixels_mask
        ]

        r = image[:, :, 2][
            pixels_mask
        ]

        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        mean_r = np.mean(r)
        mean_g = np.mean(g)
        mean_b = np.mean(b)

        std_r = np.std(r)
        std_g = np.std(g)
        std_b = np.std(b)

        # ----------------------------------------------------
        # HSV
        # ----------------------------------------------------

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        h = hsv[:, :, 0][
            pixels_mask
        ]

        s = hsv[:, :, 1][
            pixels_mask
        ]

        v = hsv[:, :, 2][
            pixels_mask
        ]

        # ----------------------------------------------------
        # Lab
        # ----------------------------------------------------

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB
        )

        L = lab[:, :, 0][
            pixels_mask
        ]

        a = lab[:, :, 1][
            pixels_mask
        ]

        lab_b = lab[:, :, 2][
            pixels_mask
        ]

        # ----------------------------------------------------
        # 色比率
        #
        # 明るさだけではなく、
        # RGBのバランスを見る。
        # ----------------------------------------------------

        total = (
            r.astype(np.float32)
            + g.astype(np.float32)
            + b.astype(np.float32)
            + 1e-6
        )

        r_ratio = np.mean(
            r / total
        )

        g_ratio = np.mean(
            g / total
        )

        b_ratio = np.mean(
            b / total
        )

        # ----------------------------------------------------
        # 色差
        # ----------------------------------------------------

        chroma = np.sqrt(
            (
                a.astype(
                    np.float32
                )
                - 128.0
            ) ** 2
            +
            (
                lab_b.astype(
                    np.float32
                )
                - 128.0
            ) ** 2
        )

        return {

            "mean_R":
                float(mean_r),

            "mean_G":
                float(mean_g),

            "mean_B":
                float(mean_b),

            "std_R":
                float(std_r),

            "std_G":
                float(std_g),

            "std_B":
                float(std_b),

            "mean_H":
                float(np.mean(h)),

            "mean_S":
                float(np.mean(s)),

            "mean_V":
                float(np.mean(v)),

            "std_S":
                float(np.std(s)),

            "std_V":
                float(np.std(v)),

            "mean_L":
                float(np.mean(L)),

            "mean_a":
                float(np.mean(a)),

            "mean_b":
                float(np.mean(lab_b)),

            "std_L":
                float(np.std(L)),

            "std_a":
                float(np.std(a)),

            "std_b":
                float(np.std(lab_b)),

            "mean_chroma":
                float(np.mean(chroma)),

            "R_ratio":
                float(r_ratio),

            "G_ratio":
                float(g_ratio),

            "B_ratio":
                float(b_ratio),

            "fundus_pixels":
                int(np.count_nonzero(
                    pixels_mask
                )),
        }

    # --------------------------------------------------------
    # 全画像統計からIQR範囲を計算
    # --------------------------------------------------------

    def calculate_iqr_bounds(
        self,
        df
    ):

        bounds = {}

        for feature in self.features:

            values = pd.to_numeric(
                df[feature],
                errors="coerce"
            ).dropna()

            if len(values) == 0:

                bounds[feature] = {
                    "q1": np.nan,
                    "q3": np.nan,
                    "iqr": np.nan,
                    "lower": np.nan,
                    "upper": np.nan,
                }

                continue

            q1 = values.quantile(
                0.25
            )

            q3 = values.quantile(
                0.75
            )

            iqr = q3 - q1

            lower = (
                q1
                - self.iqr_multiplier
                * iqr
            )

            upper = (
                q3
                + self.iqr_multiplier
                * iqr
            )

            bounds[feature] = {

                "q1":
                    float(q1),

                "q3":
                    float(q3),

                "iqr":
                    float(iqr),

                "lower":
                    float(lower),

                "upper":
                    float(upper),
            }

        return bounds

    # --------------------------------------------------------
    # 外れ値判定
    # --------------------------------------------------------

    def detect_outliers(
        self,
        df
    ):

        bounds = (
            self.calculate_iqr_bounds(
                df
            )
        )

        result = df.copy()

        outlier_columns = []

        for feature in self.features:

            column_name = (
                "outlier_"
                + feature
            )

            outlier_columns.append(
                column_name
            )

            lower = bounds[
                feature
            ]["lower"]

            upper = bounds[
                feature
            ]["upper"]

            result[
                column_name
            ] = (

                (
                    result[feature]
                    < lower
                )
                |
                (
                    result[feature]
                    > upper
                )

            )

        # ----------------------------------------------------
        # 外れ値特徴量数
        # ----------------------------------------------------

        result[
            "color_outlier_feature_count"
        ] = result[
            outlier_columns
        ].sum(axis=1)

        # ----------------------------------------------------
        # 最終判定
        # ----------------------------------------------------

        result[
            "color_outlier"
        ] = (

            result[
                "color_outlier_feature_count"
            ]
            >= self.outlier_feature_count

        )

        # ----------------------------------------------------
        # どの特徴で外れたか
        # ----------------------------------------------------

        def get_outlier_features(
            row
        ):

            names = []

            for feature in self.features:

                column = (
                    "outlier_"
                    + feature
                )

                if bool(
                    row[column]
                ):

                    names.append(
                        feature
                    )

            return ",".join(
                names
            )

        result[
            "outlier_features"
        ] = result.apply(
            get_outlier_features,
            axis=1
        )

        return (
            result,
            bounds
        )




def create_color_boxplots(
    df,
    bounds,
    output_dir
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    features = [

        "mean_R",
        "mean_G",
        "mean_B",

        "mean_L",
        "mean_a",
        "mean_b",

        "mean_S",
        "mean_V",
    ]

    # --------------------------------------------------------
    # 1つの図にまとめる
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18, 9)
    )

    axes = axes.ravel()

    for ax, feature in zip(
        axes,
        features
    ):

        values = df[
            feature
        ].dropna()

        ax.boxplot(
            values,
            vert=True,
            showfliers=True
        )

        ax.set_title(
            feature
        )

        ax.set_ylabel(
            "Value"
        )

        # ----------------------------------------------------
        # Q1 / median / Q3
        # ----------------------------------------------------

        q1 = values.quantile(
            0.25
        )

        median = values.quantile(
            0.50
        )

        q3 = values.quantile(
            0.75
        )

        ax.text(
            1.08,
            0.75,
            f"Q1 = {q1:.2f}\n"
            f"Median = {median:.2f}\n"
            f"Q3 = {q3:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top"
        )

    plt.suptitle(
        "Fundus Image Color Statistics - Boxplots",
        fontsize=16
    )

    plt.tight_layout()

    combined_path = os.path.join(
        output_dir,
        "color_statistics_boxplots.png"
    )

    plt.savefig(
        combined_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # 特徴量ごとの個別箱ひげ図
    # --------------------------------------------------------

    for feature in features:

        values = df[
            feature
        ].dropna()

        fig, ax = plt.subplots(
            figsize=(7, 6)
        )

        ax.boxplot(
            values,
            vert=True,
            showfliers=True
        )

        ax.set_title(
            f"{feature} Boxplot"
        )

        ax.set_ylabel(
            feature
        )

        q1 = values.quantile(
            0.25
        )

        median = values.quantile(
            0.50
        )

        q3 = values.quantile(
            0.75
        )

        lower = bounds[
            feature
        ]["lower"]

        upper = bounds[
            feature
        ]["upper"]

        ax.text(
            1.05,
            0.95,
            f"Q1      = {q1:.2f}\n"
            f"Median  = {median:.2f}\n"
            f"Q3      = {q3:.2f}\n"
            f"Lower   = {lower:.2f}\n"
            f"Upper   = {upper:.2f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top"
        )

        plt.tight_layout()

        path = os.path.join(
            output_dir,
            f"{feature}_boxplot.png"
        )

        plt.savefig(
            path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()




def create_color_scatterplot(
    df,
    output_dir
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    normal = df[
        ~df["color_outlier"]
    ]

    outlier = df[
        df["color_outlier"]
    ]

    ax.scatter(
        normal["mean_R"],
        normal["mean_G"],
        alpha=0.6,
        label="Normal"
    )

    ax.scatter(
        outlier["mean_R"],
        outlier["mean_G"],
        marker="x",
        s=70,
        label="Color Outlier"
    )

    ax.set_xlabel(
        "Mean R"
    )

    ax.set_ylabel(
        "Mean G"
    )

    ax.set_title(
        "Fundus Color Distribution: R vs G"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "RGB_scatter.png"
        ),
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Lab a / b
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    ax.scatter(
        normal["mean_a"],
        normal["mean_b"],
        alpha=0.6,
        label="Normal"
    )

    ax.scatter(
        outlier["mean_a"],
        outlier["mean_b"],
        marker="x",
        s=70,
        label="Color Outlier"
    )

    ax.set_xlabel(
        "Mean Lab a"
    )

    ax.set_ylabel(
        "Mean Lab b"
    )

    ax.set_title(
        "Fundus Color Distribution: Lab a vs b"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "Lab_scatter.png"
        ),
        dpi=200
    )

    plt.close()




def save_outlier_previews(
    df,
    image_dir,
    output_dir
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    outliers = df[
        df["color_outlier"]
    ]

    for _, row in outliers.iterrows():

        filename = row["image"]

        path = os.path.join(
            image_dir,
            filename
        )

        image = cv2.imread(
            path
        )

        if image is None:

            continue

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        fig, ax = plt.subplots(
            figsize=(7, 7)
        )

        ax.imshow(
            image_rgb
        )

        ax.axis("off")

        title = (
            f"{filename}\n"
            f"Outlier features: "
            f"{row['color_outlier_feature_count']}\n"
            f"{row['outlier_features']}"
        )

        ax.set_title(
            title,
            fontsize=10
        )

        output_path = os.path.join(
            output_dir,
            os.path.splitext(
                filename
            )[0]
            + "_COLOR_OUTLIER.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()





# ============================================================
# 2本目: 血管抽出ロジック
# ============================================================


def mask_touches_border(mask):
    return bool(
        np.any(mask[0, :] > 0)
        or np.any(mask[-1, :] > 0)
        or np.any(mask[:, 0] > 0)
        or np.any(mask[:, -1] > 0)
    )




def normalize_response(response, mask, low=1.0, high=99.0):
    output = np.zeros(response.shape, dtype=np.uint8)
    valid = (mask > 0) & np.isfinite(response) & (response > 0)
    values = response[valid]
    if values.size == 0:
        return output

    lo = float(np.percentile(values, low))
    hi = float(np.percentile(values, high))
    if hi <= lo + 1e-8:
        hi = float(np.max(values))
    if hi <= lo + 1e-8:
        return output

    normalized = (response.astype(np.float32) - lo) / (hi - lo)
    normalized = np.clip(normalized, 0.0, 1.0)
    output = (normalized * 255.0).astype(np.uint8)
    return mask_image(output, mask)




def positive_percentile(response, mask, percentile, default=0.0):
    valid = (mask > 0) & np.isfinite(response) & (response > 0)
    values = response[valid]
    if values.size == 0:
        return float(default)
    return float(np.percentile(values, percentile))




def fill_outside_with_median(image, mask):
    result = image.astype(np.float32).copy()
    valid = mask > 0
    if not np.any(valid):
        return result
    result[~valid] = float(np.median(result[valid]))
    return result




def suppress_near_border(response, mask, width=BORDER_SUPPRESSION_WIDTH):
    if width <= 0:
        return mask_image(response, mask)
    padded = cv2.copyMakeBorder(
        mask, width + 2, width + 2, width + 2, width + 2,
        cv2.BORDER_CONSTANT, value=0
    )
    distance_padded = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    p = width + 2
    distance = distance_padded[p:p + mask.shape[0], p:p + mask.shape[1]]
    falloff = np.clip(distance / max(float(width), 1.0), 0.0, 1.0)
    result = response.astype(np.float32) * falloff
    result[mask == 0] = 0.0
    return np.clip(result, 0, 255).astype(np.uint8)




def keep_largest_component(binary):
    binary = np.where(binary > 0, 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if count <= 1:
        return binary
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return np.where(labels == largest, 255, 0).astype(np.uint8)




def create_effective_mask(fundus_mask):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            odd_size(EFFECTIVE_MASK_ERODE_SIZE),
            odd_size(EFFECTIVE_MASK_ERODE_SIZE),
        ),
    )
    effective = safe_erode(fundus_mask, kernel)
    if np.count_nonzero(effective) == 0:
        return fundus_mask.copy()
    return np.where(effective > 0, 255, 0).astype(np.uint8)




def estimate_noise_std(green, mask):
    laplacian = cv2.Laplacian(green, cv2.CV_64F, ksize=3)
    values = laplacian[mask > 0]
    return float(np.std(values)) if values.size else 0.0




def _frangi_gamma_fallback(image, sigma):
    """gamma=None に対応していない古いscikit-image向けの互換処理。

    現行のFrangi実装で gamma=None が行う「Hessian norm の最大値の
    半分」を、1スケール分について明示的に計算する。
    """
    try:
        hessian = hessian_matrix(
            image,
            sigma=sigma,
            mode="reflect",
        )
        eigenvalues = hessian_matrix_eigvals(hessian)
        s = np.sqrt(
            np.sum(
                np.asarray(eigenvalues, dtype=np.float64) ** 2,
                axis=0,
            )
        )
        gamma = float(np.max(s)) / 2.0
    except Exception:
        # 最終的な保険。正規化画像なので極端に大きな固定値にはしない。
        gamma = 0.5

    if not np.isfinite(gamma) or gamma <= 1e-12:
        gamma = 1.0

    return gamma


def _run_frangi_compat(image, sigma):
    """scikit-imageのバージョン差を吸収してFrangiを実行する。"""
    try:
        return frangi(
            image,
            sigmas=(sigma,),
            alpha=FRANGI_ALPHA,
            beta=FRANGI_BETA,
            gamma=FRANGI_GAMMA,
            black_ridges=True,
        )
    except TypeError as exc:
        message = str(exc)
        if "NoneType" not in message or "power" not in message and "operand" not in message:
            raise

        gamma = _frangi_gamma_fallback(image, sigma)
        return frangi(
            image,
            sigmas=(sigma,),
            alpha=FRANGI_ALPHA,
            beta=FRANGI_BETA,
            gamma=gamma,
            black_ridges=True,
        )


def build_frangi_response(processed_green, mask):
    image = fill_outside_with_median(processed_green, mask) / 255.0
    responses = []

    for sigma in FRANGI_SIGMAS:
        response = _run_frangi_compat(image, sigma)
        response = np.nan_to_num(
            response.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        responses.append(response)

    weights = (0.72, 0.90, 1.00, 1.00, 1.00, 0.92)
    weighted = [r * w for r, w in zip(responses, weights)]
    response = np.maximum.reduce(weighted)
    response[mask == 0] = 0.0

    normalized = normalize_response(response, mask, low=1.0, high=99.5)
    return suppress_near_border(normalized, mask)




def build_oriented_blackhat(processed_green, mask):
    image = fill_outside_with_median(processed_green, mask).astype(np.uint8)
    response = np.zeros_like(image, dtype=np.float32)

    for length in ORIENTED_LINE_LENGTHS:
        for angle in range(0, 180, ORIENTED_LINE_ANGLE_STEP):
            kernel = np.zeros((length, length), dtype=np.uint8)
            center = length // 2
            radians = np.deg2rad(angle)
            dx = int(round(np.cos(radians) * center))
            dy = int(round(np.sin(radians) * center))
            cv2.line(
                kernel,
                (center - dx, center - dy),
                (center + dx, center + dy),
                1,
                1,
            )
            current = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
            response = np.maximum(response, current.astype(np.float32))

    for kernel_size in BLACKHAT_KERNEL_SIZES:
        size = odd_size(kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        current = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        response = np.maximum(response, current.astype(np.float32))

    response[mask == 0] = 0.0
    normalized = normalize_response(response, mask)
    return suppress_near_border(normalized, mask)




def build_dark_line_response(processed_green, mask):
    image = fill_outside_with_median(processed_green, mask)
    background = cv2.GaussianBlur(
        image, (0, 0), DARK_LINE_SIGMA, DARK_LINE_SIGMA
    )
    response = np.maximum(background - image, 0.0)
    response[mask == 0] = 0.0
    normalized = normalize_response(response, mask)
    return suppress_near_border(normalized, mask)




def build_coherence(processed_green, mask, condition):
    image = fill_outside_with_median(processed_green, mask) / 255.0
    image = image.astype(np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    sigma = (
        COHERENCE_SIGMA_NORMAL * (1.0 - condition)
        + COHERENCE_SIGMA_DIFFICULT * condition
    )
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma, sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma, sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma, sigma)

    delta = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy ** 2, 0.0))
    l1 = 0.5 * (jxx + jyy + delta)
    l2 = 0.5 * (jxx + jyy - delta)
    coherence = (l1 - l2) / (l1 + l2 + 1e-6)

    energy = np.sqrt(np.maximum(l1, 0.0))
    coherence *= np.clip(energy / 0.012, 0.0, 1.0)
    coherence = np.clip(coherence, 0.0, 1.0)
    coherence[mask == 0] = 0.0
    return coherence.astype(np.float32)




def build_local_relative_frangi(frangi_response, mask):
    response = frangi_response.astype(np.float32)
    mask_float = (mask > 0).astype(np.float32)
    local_sum = cv2.GaussianBlur(
        response * mask_float,
        (0, 0),
        LOCAL_RELATIVE_SIGMA,
        LOCAL_RELATIVE_SIGMA,
    )
    local_count = cv2.GaussianBlur(
        mask_float,
        (0, 0),
        LOCAL_RELATIVE_SIGMA,
        LOCAL_RELATIVE_SIGMA,
    )
    local_mean = local_sum / np.maximum(local_count, 1e-3)
    local_mean = np.maximum(local_mean, LOCAL_RELATIVE_MEAN_FLOOR)
    ratio = response / local_mean
    ratio[mask == 0] = 0.0
    return ratio.astype(np.float32)




def build_ring_artifact_score(processed_green, mask):
    valid = mask > 0
    ys, xs = np.nonzero(valid)
    if len(xs) < 100:
        return np.zeros_like(processed_green, dtype=np.float32)

    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    radius = np.sqrt(dx * dx + dy * dy)
    radial_x = dx / np.maximum(radius, 1e-6)
    radial_y = dy / np.maximum(radius, 1e-6)

    image = fill_outside_with_median(processed_green, mask) / 255.0
    image = image.astype(np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx * gx + gy * gy)

    radial_alignment = np.abs(gx * radial_x + gy * radial_y) / np.maximum(
        magnitude, 1e-6
    )
    reference = float(np.percentile(magnitude[valid], 75.0))
    strength = np.clip(magnitude / max(reference, 1e-6), 0.0, 1.0)
    score = radial_alignment * strength

    maximum_radius = float(np.max(radius[valid]))
    radius_ratio = radius / max(maximum_radius, 1.0)
    score[(radius_ratio < 0.10) | (radius_ratio > 1.02) | (~valid)] = 0.0
    return np.clip(score, 0.0, 1.0).astype(np.float32)




def geodesic_hysteresis(strong, candidate, mask):
    result = np.where((strong > 0) & (mask > 0), 255, 0).astype(np.uint8)
    allowed = np.where((candidate > 0) & (mask > 0), 255, 0).astype(np.uint8)
    allowed[result > 0] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for _ in range(HYSTERESIS_MAX_ITERATIONS):
        expanded = cv2.dilate(result, kernel, iterations=1)
        expanded = cv2.bitwise_and(expanded, allowed)
        if np.array_equal(expanded, result):
            break
        result = expanded

    return mask_image(result, mask)




def threshold_vessels(
    frangi_response,
    blackhat,
    dark_line,
    coherence,
    local_relative,
    ring_score,
    mask,
    condition,
):
    valid = mask > 0
    if not np.any(valid):
        zero = np.zeros_like(mask)
        return zero, zero.copy(), zero.copy(), 0.0, 0.0

    f = frangi_response.astype(np.float32)
    b = blackhat.astype(np.float32)
    d = dark_line.astype(np.float32)
    c = coherence.astype(np.float32)

    boost = 4.0 * condition
    f_high = positive_percentile(
        f, mask, min(99.0, STRONG_FRANGI_PERCENTILE + boost)
    )
    f_low = positive_percentile(
        f, mask, min(98.0, WEAK_FRANGI_PERCENTILE + boost)
    )
    b_threshold = positive_percentile(
        b, mask, min(98.0, BLACKHAT_PERCENTILE + boost)
    )
    d_threshold = positive_percentile(
        d, mask, min(98.0, DARK_PERCENTILE + boost)
    )

    f_low = max(f_low, 5.0, f_high * (0.24 + 0.06 * condition))
    f_high = max(f_high, f_low + 2.0)
    b_threshold = max(b_threshold, 5.0)
    d_threshold = max(d_threshold, 5.0)

    line_support = b >= b_threshold
    dark_support = d >= d_threshold
    coherence_strong = c >= (COHERENCE_STRONG_MIN + 0.06 * condition)
    coherence_weak = c >= (COHERENCE_WEAK_MIN + 0.05 * condition)

    # FrangiをStrong seedの必須条件とし、暗模様だけのseed化を防ぐ。
    strong = (
        (f >= f_high)
        & coherence_strong
        & (line_support | dark_support)
        & valid
    )
    strong |= (
        (f >= max(245.0, f_high * 1.25))
        & (c >= 0.28)
        & valid
    )

    support_count = (
        line_support.astype(np.uint8)
        + dark_support.astype(np.uint8)
        + coherence_weak.astype(np.uint8)
    )
    weak = (f >= f_low) & (support_count >= 2) & valid

    local_rescue = (
        (local_relative >= LOCAL_RELATIVE_MIN)
        & (f >= max(5.0, f_low * 0.60))
        & coherence_weak
        & (line_support | dark_support)
        & valid
    )

    candidate = weak | local_rescue | strong

    distance = cv2.distanceTransform(
        np.where(mask > 0, 255, 0).astype(np.uint8), cv2.DIST_L2, 5
    )
    border_safe = distance >= float(BORDER_SUPPRESSION_WIDTH)
    candidate &= border_safe
    strong &= border_safe

    if (
        RING_PENALTY_ENABLED
        and condition >= RING_CONDITION_MIN
        and ring_score is not None
    ):
        ring_bad = ring_score >= RING_SCORE_THRESHOLD
        candidate &= ~(ring_bad & (f < f_high * 1.15))
        strong &= ~(ring_bad & (f < f_high * 1.05))

    strong_image = (strong.astype(np.uint8) * 255)
    candidate_image = (candidate.astype(np.uint8) * 255)
    vessel_raw = geodesic_hysteresis(strong_image, candidate_image, mask)
    vessel_raw[strong] = 255

    return vessel_raw, strong_image, candidate_image, f_high, f_low




def remove_small_vessel_components(binary, frangi_response, mask):
    binary = mask_image(binary, mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    output = np.zeros_like(binary)

    for label in range(1, count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 2:
            continue

        component_skeleton = skeletonize(component)
        length = int(np.count_nonzero(component_skeleton))
        mean_frangi = float(np.mean(frangi_response[component]))

        if (
            area < MIN_COMPONENT_AREA
            and length < MIN_COMPONENT_LENGTH
            and mean_frangi < MIN_COMPONENT_MEAN_FRANGI
        ):
            continue

        output[component] = 255

    return mask_image(output, mask)




def make_skeleton(vessel_binary, mask):
    vessel = mask_image(vessel_binary, mask)
    if CENTERLINE_CLOSE_SIZE > 1:
        size = odd_size(CENTERLINE_CLOSE_SIZE)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        vessel = safe_close(vessel, kernel)
        vessel = mask_image(vessel, mask)

    skeleton = skeletonize(vessel > 0)
    return mask_image((skeleton.astype(np.uint8) * 255), mask)




def count_endpoints_and_branchpoints(skeleton):
    binary = (skeleton > 0).astype(np.uint8)
    padded = np.pad(binary, 1, mode="constant")
    neighbors = np.zeros_like(binary, dtype=np.uint8)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbors += padded[
                1 + dy:1 + dy + binary.shape[0],
                1 + dx:1 + dx + binary.shape[1],
            ]

    endpoints = int(np.count_nonzero((binary > 0) & (neighbors == 1)))
    branchpoints = int(np.count_nonzero((binary > 0) & (neighbors >= 3)))
    return endpoints, branchpoints




def fractal_dimension(binary, mask=None):
    data = binary > 0
    if mask is not None:
        data &= mask > 0
    if np.count_nonzero(data) < 10:
        return 0.0

    minimum_side = min(data.shape)
    power_size = 2 ** int(np.floor(np.log2(minimum_side)))
    if power_size < 4:
        return 0.0

    y0 = (data.shape[0] - power_size) // 2
    x0 = (data.shape[1] - power_size) // 2
    data = data[y0:y0 + power_size, x0:x0 + power_size]

    sizes = 2 ** np.arange(int(np.log2(power_size)), 1, -1)
    counts = []

    for size in sizes:
        reduced = np.add.reduceat(
            np.add.reduceat(data, np.arange(0, power_size, size), axis=0),
            np.arange(0, power_size, size),
            axis=1,
        )
        counts.append(int(np.count_nonzero(reduced)))

    sizes = np.asarray(sizes, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    valid = counts > 0
    if np.count_nonzero(valid) < 2:
        return 0.0

    coefficient = np.polyfit(
        np.log(sizes[valid]), np.log(counts[valid]), 1
    )
    return float(-coefficient[0])




def create_overlay(original, skeleton, mask, thickness):
    line = mask_image(skeleton, mask)
    thickness = max(1, int(thickness))
    if thickness > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (odd_size(thickness), odd_size(thickness))
        )
        line = cv2.dilate(line, kernel, iterations=1)
        line = mask_image(line, mask)

    output = original.astype(np.float32).copy()
    active = line > 0
    red = np.zeros_like(original, dtype=np.float32)
    red[:, :, 2] = 255.0
    output[active] = (
        output[active] * (1.0 - OVERLAY_ALPHA)
        + red[active] * OVERLAY_ALPHA
    )
    return np.clip(output, 0, 255).astype(np.uint8)




def build_final_output(original, skeleton, mask):
    if OUTPUT_IMAGE_MODE == "overlay":
        return create_overlay(
            original, skeleton, mask, OUTPUT_LINE_THICKNESS
        )
    if OUTPUT_IMAGE_MODE == "vessel_only":
        return mask_image(skeleton, mask)
    raise ValueError(
        "OUTPUT_IMAGE_MODE は 'overlay' または 'vessel_only' にしてください。"
    )




def colorize_response(image):
    if image.dtype != np.uint8:
        finite = np.nan_to_num(image.astype(np.float32))
        if float(np.max(finite)) <= 1.0:
            image = np.clip(finite * 255.0, 0, 255).astype(np.uint8)
        else:
            image = np.clip(finite, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()




def prepare_debug_tile(title, image, tile_size=DEBUG_TILE_SIZE):
    """デバッグ用タイルを作成する。"""
    visual = colorize_response(image)
    visual = cv2.resize(visual, tile_size, interpolation=cv2.INTER_AREA)

    cv2.rectangle(
        visual,
        (0, 0),
        (tile_size[0], DEBUG_TITLE_HEIGHT),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        visual,
        title,
        (5, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return visual


def build_debug_montage(preprocess_steps, extraction_items):
    """
    1枚のデバッグ画像を作る。

    01〜15:
        前処理の各ステップ + 血管抽出の各ステップを通常サイズで配置。
    16 final overlay:
        最後に横幅いっぱいで大きく配置。

    FINAL VESSEL EXTRACTION と ORIGINAL FUNDUS は保存しない。
    """
    items = []

    for title, image in preprocess_steps.items():
        items.append((title, image))

    items.extend(extraction_items)

    if not items:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    # 最後の「16 final overlay」だけを大きくする。
    # それ以外は従来どおりのタイルサイズで4列配置する。
    normal_items = items[:-1]
    final_title, final_image = items[-1]

    tiles = [
        prepare_debug_tile(title, image)
        for title, image in normal_items
    ]

    columns = DEBUG_COLUMNS
    rows = int(np.ceil(len(tiles) / columns)) if tiles else 0

    width = columns * DEBUG_TILE_SIZE[0]
    top_height = rows * DEBUG_TILE_SIZE[1]

    # final overlay は横幅いっぱい・高さ2倍で表示する。
    final_tile_size = (
        width,
        DEBUG_TILE_SIZE[1] * 2,
    )
    final_tile = prepare_debug_tile(
        final_title,
        final_image,
        tile_size=final_tile_size,
    )

    total_height = top_height + final_tile_size[1]
    montage = np.zeros(
        (total_height, width, 3),
        dtype=np.uint8,
    )

    for index, tile in enumerate(tiles):
        row = index // columns
        column = index % columns

        y0 = row * DEBUG_TILE_SIZE[1]
        x0 = column * DEBUG_TILE_SIZE[0]

        montage[
            y0:y0 + DEBUG_TILE_SIZE[1],
            x0:x0 + DEBUG_TILE_SIZE[0],
        ] = tile

    montage[
        top_height:top_height + final_tile_size[1],
        0:width,
    ] = final_tile

    return montage


# ============================================================
# 統合処理: 1枚の画像
# ============================================================

def process_one_image(
    original,
    green_preprocessed,
    fundus_mask,
):
    effective_mask = create_effective_mask(fundus_mask)

    # --------------------------------------------------------
    # 血管特徴量
    # --------------------------------------------------------

    frangi_response = build_frangi_response(
        green_preprocessed,
        effective_mask,
    )

    blackhat = build_oriented_blackhat(
        green_preprocessed,
        effective_mask,
    )

    dark_line = build_dark_line_response(
        green_preprocessed,
        effective_mask,
    )

    # 画像状態は1本目の前処理後画像から評価する
    condition_info = {
        "mean": float(np.mean(green_preprocessed[effective_mask > 0]))
        if np.any(effective_mask > 0) else 0.0,
        "median": float(np.median(green_preprocessed[effective_mask > 0]))
        if np.any(effective_mask > 0) else 0.0,
        "contrast": 0.0,
        "noise_std": estimate_noise_std(
            green_preprocessed,
            effective_mask,
        ),
    }

    if np.any(effective_mask > 0):
        pixels = green_preprocessed[effective_mask > 0]
        condition_info["contrast"] = float(
            max(
                np.percentile(pixels, 90.0)
                - np.percentile(pixels, 10.0),
                1.0,
            )
        )

    condition = 0.0

    coherence = build_coherence(
        green_preprocessed,
        effective_mask,
        condition,
    )

    local_relative = build_local_relative_frangi(
        frangi_response,
        effective_mask,
    )

    ring_score = build_ring_artifact_score(
        green_preprocessed,
        effective_mask,
    )

    (
        vessel_raw,
        strong_seed,
        candidate,
        high_threshold,
        low_threshold,
    ) = threshold_vessels(
        frangi_response,
        blackhat,
        dark_line,
        coherence,
        local_relative,
        ring_score,
        effective_mask,
        condition,
    )

    vessel_clean = remove_small_vessel_components(
        vessel_raw,
        frangi_response,
        effective_mask,
    )

    skeleton = make_skeleton(
        vessel_clean,
        effective_mask,
    )

    final_image = build_final_output(
        original,
        skeleton,
        fundus_mask,
    )

    debug_overlay = create_overlay(
        original,
        skeleton,
        fundus_mask,
        DEBUG_OVERLAY_LINE_THICKNESS,
    )

    local_relative_visual = np.clip(
        local_relative
        / max(LOCAL_RELATIVE_MIN * 2.0, 1e-6)
        * 255.0,
        0,
        255,
    ).astype(np.uint8)

    debug_items = [
        ("01 original fundus", original),
        ("02 green preprocessed", green_preprocessed),
        ("03 fundus mask", fundus_mask),
        ("04 effective mask", effective_mask),
        ("05 frangi", frangi_response),
        ("06 oriented blackhat", blackhat),
        ("07 dark line", dark_line),
        (
            "08 coherence",
            np.clip(
                coherence * 255.0,
                0,
                255,
            ).astype(np.uint8),
        ),
        ("09 local relative", local_relative_visual),
        (
            "10 ring score",
            np.clip(
                ring_score * 255.0,
                0,
                255,
            ).astype(np.uint8),
        ),
        ("11 strong seed", strong_seed),
        ("12 candidate", candidate),
        ("13 vessel raw", vessel_raw),
        ("14 vessel clean", vessel_clean),
        ("15 skeleton", skeleton),
        ("16 final overlay", debug_overlay),
    ]

    endpoints, branchpoints = count_endpoints_and_branchpoints(
        skeleton
    )

    fundus_pixels = max(
        1,
        int(np.count_nonzero(fundus_mask)),
    )
    effective_pixels = max(
        1,
        int(np.count_nonzero(effective_mask)),
    )
    vessel_pixels = int(
        np.count_nonzero(vessel_clean)
    )
    skeleton_pixels = int(
        np.count_nonzero(skeleton)
    )

    metrics = {
        "fundus_pixels": fundus_pixels,
        "effective_mask_pixels": effective_pixels,
        "vessel_pixels": vessel_pixels,
        "skeleton_pixels": skeleton_pixels,
        "vessel_ratio": vessel_pixels / fundus_pixels,
        "skeleton_ratio": skeleton_pixels / fundus_pixels,
        "vessel_ratio_effective": vessel_pixels / effective_pixels,
        "skeleton_ratio_effective": skeleton_pixels / effective_pixels,
        "fractal_dimension": fractal_dimension(
            skeleton,
            effective_mask,
        ),
        "endpoint_count": endpoints,
        "branchpoint_count": branchpoints,
        "green_mean_after_preprocess": condition_info["mean"],
        "green_median_after_preprocess": condition_info["median"],
        "green_contrast_after_preprocess": condition_info["contrast"],
        "green_noise_std_after_preprocess": condition_info["noise_std"],
        "frangi_high_threshold": high_threshold,
        "frangi_low_threshold": low_threshold,
        "mask_touches_border": bool(
            np.any(fundus_mask[0, :] > 0)
            or np.any(fundus_mask[-1, :] > 0)
            or np.any(fundus_mask[:, 0] > 0)
            or np.any(fundus_mask[:, -1] > 0)
        ),
    }

    return final_image, debug_items, metrics


# ============================================================
# 色統計
# ============================================================

def analyze_all_colors(files, image_dir, preprocessor):
    analyzer = ColorAnalyzer(
        iqr_multiplier=COLOR_IQR_MULTIPLIER,
        outlier_feature_count=COLOR_OUTLIER_FEATURE_COUNT,
    )

    records = []

    print()
    print("=" * 72)
    print("STEP 1: 色統計を計算します")
    print("=" * 72)

    for filename in tqdm(files, desc="Color statistics"):
        path = os.path.join(image_dir, filename)
        image = cv2.imread(path, cv2.IMREAD_COLOR)

        if image is None:
            print(f"[SKIP] 画像を読み込めません: {path}")
            continue

        mask = preprocessor.create_fundus_mask(image)
        statistics = analyzer.calculate_color_statistics(
            image,
            mask,
        )

        if statistics is None:
            print(f"[SKIP] 眼底領域を取得できません: {filename}")
            continue

        statistics["image"] = filename
        statistics["mask_touches_border"] = (
            preprocessor.mask_touches_border(mask)
        )
        records.append(statistics)

    if not records:
        raise RuntimeError(
            "色統計を計算できる画像がありません。"
        )

    dataframe = pd.DataFrame(records)
    return dataframe


# ============================================================
# 出力フォルダを空にする
# ============================================================

def clear_output_directory(directory):
    os.makedirs(directory, exist_ok=True)

    for name in os.listdir(directory):
        path = os.path.join(directory, name)

        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


# ============================================================
# メイン処理
# ============================================================

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not os.path.isdir(SOURCE_IMAGE_DIR):
        raise FileNotFoundError(
            f"入力フォルダがありません: {SOURCE_IMAGE_DIR}"
        )

    files = sorted(
        filename
        for filename in os.listdir(SOURCE_IMAGE_DIR)
        if os.path.isfile(
            os.path.join(SOURCE_IMAGE_DIR, filename)
        )
        and os.path.splitext(filename)[1].lower()
        in SUPPORTED_EXTENSIONS
    )

    if not files:
        raise FileNotFoundError(
            f"{SOURCE_IMAGE_DIR} に対応画像がありません。"
        )

    # --------------------------------------------------------
    # 実行開始時に、今回使う出力先だけ完全に空にする
    # --------------------------------------------------------

    output_dirs = [
        # 通常の血管抽出結果
        VESSEL_OUTPUT_DIR,
        # 先頭20件の統合Debug画像
        DEBUG_MONTAGE_DIR,
        # カラー判定の根拠資料
        COLOR_VISUALIZATION_DIR,
    ]

    if CLEAR_OUTPUTS_AT_START:
        for directory in output_dirs:
            clear_output_directory(directory)

        for csv_path in [
            COLOR_STATISTICS_CSV,
            COLOR_OUTLIERS_CSV,
        ]:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    preprocessor = FundusPreprocessor(
        image_dir=SOURCE_IMAGE_DIR,
        debug_sample_count=DEBUG_SAMPLE_COUNT,
        random_seed=RANDOM_SEED,
    )

    color_analyzer = ColorAnalyzer(
        iqr_multiplier=COLOR_IQR_MULTIPLIER,
        outlier_feature_count=COLOR_OUTLIER_FEATURE_COUNT,
    )

    # --------------------------------------------------------
    # STEP 1: 色統計
    #
    # カラーで外れ値を判定する根拠を残すため、
    # 統計CSV・箱ひげ図・散布図・外れ値プレビューを保存する。
    # --------------------------------------------------------

    color_df = analyze_all_colors(
        files,
        SOURCE_IMAGE_DIR,
        preprocessor,
    )

    color_df, color_bounds = color_analyzer.detect_outliers(
        color_df
    )

    color_df.to_csv(
        COLOR_STATISTICS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    outlier_df = color_df[
        color_df["color_outlier"]
    ].copy()

    outlier_df.to_csv(
        COLOR_OUTLIERS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # カラー判定の根拠となる図を保存
    create_color_boxplots(
        color_df,
        color_bounds,
        COLOR_BOXPLOT_DIR,
    )

    create_color_scatterplot(
        color_df,
        COLOR_VISUALIZATION_DIR,
    )

    save_outlier_previews(
        color_df,
        SOURCE_IMAGE_DIR,
        COLOR_OUTLIER_PREVIEW_DIR,
    )

    files_to_process = [
        filename
        for filename in files
        if filename in set(color_df["image"].tolist())
    ]

    color_lookup = (
        color_df.set_index("image").to_dict("index")
    )

    # 「冒頭20組」なので、ランダム抽出ではなく処理対象の先頭20件。
    debug_files = set(
        files_to_process[:DEBUG_SAMPLE_COUNT]
    )

    records = []
    debug_count = 0
    skipped_count = 0

    print()
    print("=" * 72)
    print("STEP 2: 前処理・血管抽出")
    print("=" * 72)

    for filename in tqdm(
        files_to_process,
        desc="Preprocess + Vessel extraction",
    ):
        image_path = os.path.join(
            SOURCE_IMAGE_DIR,
            filename,
        )

        original = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR,
        )

        if original is None:
            print(
                f"[SKIP] 画像を読み込めません: {image_path}"
            )
            skipped_count += 1
            continue

        try:
            (
                green_preprocessed,
                _color_preprocessed,
                fundus_mask,
                preprocess_steps,
                before_metrics,
                after_metrics,
                diagnostics,
            ) = preprocessor.preprocess_one_image(
                original
            )

            final_image, debug_items, metrics = (
                process_one_image(
                    original,
                    green_preprocessed,
                    fundus_mask,
                )
            )

        except Exception as exc:
            print(
                f"[SKIP] 処理に失敗しました: "
                f"{filename} ({type(exc).__name__}: {exc})"
            )
            skipped_count += 1
            continue

        image_id = os.path.splitext(filename)[0]

        # ----------------------------------------------------
        # 通常出力
        #
        # 必要なのは血管抽出画像だけなので、1画像につき1ファイル。
        # ----------------------------------------------------

        vessel_path = os.path.join(
            VESSEL_OUTPUT_DIR,
            image_id + "_vessel.png",
        )

        if not cv2.imwrite(vessel_path, final_image):
            print(
                f"[WARN] 血管抽出画像を保存できませんでした: "
                f"{vessel_path}"
            )

        # ----------------------------------------------------
        # Debug
        #
        # 先頭20件だけ、
        #   ・前処理の各ステップ
        #   ・血管抽出の各ステップ
        #   ・16 final overlay
        # を1枚にまとめて保存する。
        #
        # FINAL VESSEL EXTRACTION と ORIGINAL FUNDUS は含めない。
        # 個々の中間画像も一切保存しない。
        # ----------------------------------------------------

        if filename in debug_files:
            debug_count += 1

            montage = build_debug_montage(
                preprocess_steps,
                debug_items,
            )

            debug_path = os.path.join(
                DEBUG_MONTAGE_DIR,
                f"{debug_count:02d}_{image_id}_debug.jpg",
            )

            if not cv2.imwrite(
                debug_path,
                montage,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            ):
                print(
                    f"[WARN] Debug画像を保存できませんでした: "
                    f"{debug_path}"
                )

        # ----------------------------------------------------
        # 指標はメモリ上だけで保持する。
        # CSVは保存しない。
        # ----------------------------------------------------

        color_info = color_lookup.get(
            filename,
            {},
        )

        record = {
            "image": filename,
            "color_outlier": bool(
                color_info.get(
                    "color_outlier",
                    False,
                )
            ),
            "color_outlier_feature_count": int(
                color_info.get(
                    "color_outlier_feature_count",
                    0,
                )
            ),
            "outlier_features": color_info.get(
                "outlier_features",
                "",
            ),

            "before_mean": before_metrics["mean"],
            "before_std": before_metrics["std"],
            "before_min": before_metrics["min"],
            "before_max": before_metrics["max"],
            "before_dark_ratio": before_metrics["dark_ratio"],

            "after_mean": after_metrics["mean"],
            "after_std": after_metrics["std"],
            "after_min": after_metrics["min"],
            "after_max": after_metrics["max"],
            "after_dark_ratio": after_metrics["dark_ratio"],

            "noise_std": diagnostics["noise_std"],
            "pre_denoise_h": diagnostics["pre_denoise_h"],
            "brightness_scale": diagnostics["brightness_scale"],
            "clahe_clip_limit": diagnostics["clahe_clip_limit"],
            "is_dark": diagnostics["is_dark"],
            "is_noisy": diagnostics["is_noisy"],
            "preprocess_mask_touches_border": diagnostics[
                "touches_border"
            ],
        }

        record.update(metrics)
        records.append(record)

    if not records:
        raise RuntimeError(
            "血管抽出に成功した画像がありません。"
        )

    # --------------------------------------------------------
    # 結果表示
    # --------------------------------------------------------

    dataframe = pd.DataFrame(records)

    print()
    print("=" * 72)
    print("処理が完了しました")
    print("=" * 72)
    print(f"入力画像数             : {len(files)}")
    print(f"処理対象画像数         : {len(files_to_process)}")
    print(f"血管抽出成功数         : {len(dataframe)}")
    print(f"スキップ数             : {skipped_count}")
    print(f"Debug保存数            : {debug_count}")
    print()
    print(f"血管抽出画像           : {VESSEL_OUTPUT_DIR}/")
    print(f"Debug（先頭20件）      : {DEBUG_MONTAGE_DIR}/")
    print()
    print("保存する通常画像は、血管抽出画像だけです。")
    print("前処理画像・眼底マスク・血管マスク・Skeleton・Overlay・CSVは保存しません。")
    print("=" * 72)


if __name__ == "__main__":
    main()
