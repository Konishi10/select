import os
import random
import shutil
import re

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
SOURCE_IMAGE_DIR = "healthy_images"

SUPPORTED_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
)

# ------------------------------------------------------------
# 出力
# ------------------------------------------------------------

VESSEL_OUTPUT_DIR = "vessel_extracted"

DEBUG_MONTAGE_DIR = "vessel_debug_20"

# ------------------------------------------------------------
# 色統計・外れ値
# ------------------------------------------------------------

COLOR_STATISTICS_CSV = "color_statistics.csv"
COLOR_OUTLIERS_CSV = "color_outliers.csv"

COLOR_VISUALIZATION_DIR = "color_visualization"

COLOR_BOXPLOT_DIR = os.path.join(
    COLOR_VISUALIZATION_DIR,
    "boxplots"
)

COLOR_OUTLIER_PREVIEW_DIR = os.path.join(
    COLOR_VISUALIZATION_DIR,
    "outlier_previews"
)

COLOR_IQR_MULTIPLIER = 1.5

# 4項目のうち2項目以上が外れ値なら色外れ値
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

FRANGI_SIGMAS = (
    0.8,
    1.1,
    1.5,
    2.0,
    2.8,
    3.8
)

FRANGI_ALPHA = 0.5
FRANGI_BETA = 0.5
FRANGI_GAMMA = None

ORIENTED_LINE_LENGTHS = (
    9,
    15,
    23
)

ORIENTED_LINE_ANGLE_STEP = 15

BLACKHAT_KERNEL_SIZES = (
    11,
    17,
    25,
    31
)

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

DEBUG_TILE_SIZE = (
    280,
    280
)

DEBUG_COLUMNS = 4
DEBUG_TITLE_HEIGHT = 24

# 最終overlayの表示線幅
OUTPUT_LINE_THICKNESS = 3
DEBUG_OVERLAY_LINE_THICKNESS = 5
OVERLAY_ALPHA = 0.88

# 通常出力は血管抽出画像そのもの
OUTPUT_IMAGE_MODE = "vessel_only"

CLEAR_OUTPUTS_AT_START = True


# ============================================================
# 1本目: 共通ユーティリティ・前処理・色解析
# ============================================================


def odd_size(value):
    """OpenCVのカーネル用に、指定値以上の奇数へ丸める。"""

    value = max(
        1,
        int(round(value))
    )

    return (
        value
        if value % 2 == 1
        else value + 1
    )


def safe_erode(
    mask,
    kernel,
    iterations=1
):

    kh, kw = kernel.shape[:2]

    pad = max(
        kh,
        kw
    )

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


def safe_dilate(
    mask,
    kernel,
    iterations=1
):

    kh, kw = kernel.shape[:2]

    pad = max(
        kh,
        kw
    )

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


def safe_close(
    mask,
    kernel
):

    kh, kw = kernel.shape[:2]

    pad = max(
        kh,
        kw
    )

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


def normalize_01(
    image,
    mask=None
):

    image = image.astype(
        np.float32
    )

    if mask is not None:
        values = image[
            mask > 0
        ]
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


def to_uint8(
    image
):

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


def mask_image(
    image,
    mask
):
    """画像をマスク領域だけに制限する。"""

    mask_u8 = np.where(
        mask > 0,
        255,
        0
    ).astype(
        np.uint8
    )

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


def keep_largest_component(
    binary
):
    """二値画像から最大の連結成分だけを残す。"""

    binary = np.where(
        binary > 0,
        255,
        0
    ).astype(
        np.uint8
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    if count <= 1:

        return binary

    areas = stats[
        1:,
        cv2.CC_STAT_AREA
    ]

    largest = (
        1
        + int(
            np.argmax(areas)
        )
    )

    return np.where(
        labels == largest,
        255,
        0,
    ).astype(
        np.uint8
    )


# ============================================================
# FundusPreprocessor
# ============================================================


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

    # --------------------------------------------------------
    # 眼底マスク
    # --------------------------------------------------------

    def create_fundus_mask(
        self,
        img
    ):
        """眼底画像の有効領域だけを残すマスクを作る。"""

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

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

        threshold_value = float(
            max(
                FUNDUS_THRESHOLD,
                min(
                    float(otsu_value),
                    80.0
                ),
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
                odd_size(
                    FUNDUS_CLOSE_SIZE
                ),
                odd_size(
                    FUNDUS_CLOSE_SIZE
                ),
            ),
        )

        mask = safe_close(
            mask,
            kernel
        )

        mask = keep_largest_component(
            mask
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        clean_mask = np.zeros_like(
            mask
        )

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
        # 候補領域が画像端まで広がった場合の補正
        # ----------------------------------------------------

        h, w = gray.shape[:2]

        image_area = float(
            max(
                h * w,
                1
            )
        )

        contour_area = float(
            cv2.contourArea(
                largest_contour
            )
        )

        area_ratio = (
            contour_area
            / image_area
        )

        touches_border = bool(
            np.any(
                clean_mask[0, :] > 0
            )
            or np.any(
                clean_mask[-1, :] > 0
            )
            or np.any(
                clean_mask[:, 0] > 0
            )
            or np.any(
                clean_mask[:, -1] > 0
            )
        )

        if (
            touches_border
            or area_ratio > 0.97
        ):

            ellipse_mask = np.zeros_like(
                clean_mask
            )

            if len(largest_contour) >= 5:

                try:

                    ellipse = cv2.fitEllipse(
                        largest_contour
                    )

                    (
                        cx,
                        cy
                    ), (
                        axis_a,
                        axis_b
                    ), angle = ellipse

                    axis_a = max(
                        float(axis_a) * 0.97,
                        10.0
                    )

                    axis_b = max(
                        float(axis_b) * 0.97,
                        10.0
                    )

                    cv2.ellipse(
                        ellipse_mask,
                        (
                            int(round(cx)),
                            int(round(cy)),
                        ),
                        (
                            max(
                                1,
                                int(
                                    round(
                                        axis_a / 2.0
                                    )
                                )
                            ),
                            max(
                                1,
                                int(
                                    round(
                                        axis_b / 2.0
                                    )
                                )
                            ),
                        ),
                        float(angle),
                        0,
                        360,
                        255,
                        -1,
                    )

                except cv2.error:

                    ellipse_mask = (
                        np.zeros_like(
                            clean_mask
                        )
                    )

            if np.count_nonzero(
                ellipse_mask
            ) == 0:

                center = (
                    w // 2,
                    h // 2
                )

                axes = (
                    max(
                        1,
                        int(
                            round(
                                w * 0.475
                            )
                        )
                    ),
                    max(
                        1,
                        int(
                            round(
                                h * 0.475
                            )
                        )
                    ),
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

        # 最終的に外周へ少し余裕を持たせる
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                odd_size(
                    FUNDUS_ERODE_SIZE
                ),
                odd_size(
                    FUNDUS_ERODE_SIZE
                ),
            ),
        )

        clean_mask = safe_erode(
            clean_mask,
            erode_kernel,
            iterations=1
        )

        if (
            np.any(
                clean_mask[0, :] > 0
            )
            or np.any(
                clean_mask[-1, :] > 0
            )
            or np.any(
                clean_mask[:, 0] > 0
            )
            or np.any(
                clean_mask[:, -1] > 0
            )
        ):

            border = max(
                1,
                int(
                    round(
                        min(h, w) * 0.01
                    )
                )
            )

            clean_mask[
                :border,
                :
            ] = 0

            clean_mask[
                -border:,
                :
            ] = 0

            clean_mask[
                :,
                :border
            ] = 0

            clean_mask[
                :,
                -border:
            ] = 0

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
            np.any(
                mask[0, :] > 0
            )
            or np.any(
                mask[-1, :] > 0
            )
            or np.any(
                mask[:, 0] > 0
            )
            or np.any(
                mask[:, -1] > 0
            )
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

        current_mean = np.mean(
            pixels
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


# ============================================================
# ColorAnalyzer
# ============================================================


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
        # 外れ値判定に使用する4項目
        #
        # mean_a / mean_b / mean_S は使用しない。
        # ----------------------------------------------------

        self.features = [

            "mean_R",
            "mean_G",
            "mean_B",
            "mean_L",

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
        # Lab
        #
        # Lだけ使用する。
        # a / b は計算・保存しない。
        # ----------------------------------------------------

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB
        )

        L = lab[:, :, 0][
            pixels_mask
        ]

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

            "mean_L":
                float(np.mean(L)),

            "std_L":
                float(np.std(L)),

            "fundus_pixels":
                int(
                    np.count_nonzero(
                        pixels_mask
                    )
                ),
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

            iqr = (
                q3 - q1
            )

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
        #
        # 4項目中2項目以上が外れ値
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


# ============================================================
# 色統計 箱ひげ図
# ============================================================


def create_color_boxplots(
    df,
    bounds,
    output_dir
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 使用する4項目
    # --------------------------------------------------------

    features = [

        "mean_R",
        "mean_G",
        "mean_B",
        "mean_L",

    ]

    # --------------------------------------------------------
    # 特徴量ごとのカラーマップ
    # --------------------------------------------------------

    colormap_stops = {

        "mean_R": [
            (0.0, "black"),
            (1.0, "red"),
        ],

        "mean_G": [
            (0.0, "black"),
            (1.0, "green"),
        ],

        "mean_B": [
            (0.0, "black"),
            (1.0, "blue"),
        ],

        "mean_L": [
            (0.0, "black"),
            (1.0, "white"),
        ],

    }

    def add_side_colormap(
        fig,
        ax,
        feature,
        values
    ):
        """
        箱ひげ図の右側に縦カラーマップを配置する。
        """

        cmap = (
            LinearSegmentedColormap.from_list(
                feature + "_cmap",
                colormap_stops[feature]
            )
        )

        pos = ax.get_position()

        gap = 0.008
        bar_width = 0.012

        colorbar_ax = fig.add_axes([
            pos.x1 + gap,
            pos.y0,
            bar_width,
            pos.height,
        ])

        gradient = np.linspace(
            0,
            1,
            256
        ).reshape(
            -1,
            1
        )

        colorbar_ax.imshow(
            gradient,
            aspect="auto",
            cmap=cmap,
            origin="lower",
            extent=(
                0,
                1,
                0,
                1
            ),
        )

        colorbar_ax.set_xlim(
            0,
            1
        )

        colorbar_ax.set_ylim(
            0,
            1
        )

        colorbar_ax.set_xticks([])
        colorbar_ax.set_yticks([])

        for spine in (
            colorbar_ax.spines.values()
        ):

            spine.set_visible(
                False
            )

    # --------------------------------------------------------
    # 1つの図にまとめる
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(19, 7)
    )

    axes = np.atleast_1d(
        axes
    ).ravel()

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

        # x軸の「1」は表示しない
        ax.set_xticks([])

        # ----------------------------------------------------
        # Q1 / Median / Q3
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
            1.34,
            0.75,
            f"Q1 = {q1:.2f}\n"
            f"Median = {median:.2f}\n"
            f"Q3 = {q3:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top"
        )

    fig.suptitle(
        "Fundus Image Color Statistics - Boxplots",
        fontsize=16
    )

    fig.tight_layout(
        rect=[
            0.02,
            0.03,
            0.96,
            0.94
        ],
        w_pad=3.0,
        h_pad=2.0
    )

    for ax, feature in zip(
        axes,
        features
    ):

        values = df[
            feature
        ].dropna()

        add_side_colormap(
            fig,
            ax,
            feature,
            values
        )

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

        # x軸の「1」は表示しない
        ax.set_xticks([])

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
            1.34,
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

        fig.tight_layout(
            rect=[
                0.08,
                0.06,
                0.88,
                0.96
            ]
        )

        add_side_colormap(
            fig,
            ax,
            feature,
            values
        )

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


# ============================================================
# 色散布図
# ============================================================


def create_color_scatterplot(
    df,
    output_dir
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    normal = df[
        ~df["color_outlier"]
    ]

    outlier = df[
        df["color_outlier"]
    ]

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

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


# ============================================================
# 外れ値画像プレビュー
# ============================================================


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
# 色統計
# ============================================================


def analyze_all_colors(
    files,
    image_dir,
    preprocessor
):

    analyzer = ColorAnalyzer(
        iqr_multiplier=COLOR_IQR_MULTIPLIER,
        outlier_feature_count=(
            COLOR_OUTLIER_FEATURE_COUNT
        ),
    )

    records = []

    print()
    print("=" * 72)
    print("STEP 1: 色統計を計算します")
    print("=" * 72)

    for filename in tqdm(
        files,
        desc="Color statistics"
    ):

        path = os.path.join(
            image_dir,
            filename
        )

        image = cv2.imread(
            path,
            cv2.IMREAD_COLOR
        )

        if image is None:

            print(
                f"[SKIP] 画像を読み込めません: {path}"
            )

            continue

        mask = (
            preprocessor.create_fundus_mask(
                image
            )
        )

        statistics = (
            analyzer.calculate_color_statistics(
                image,
                mask,
            )
        )

        if statistics is None:

            print(
                f"[SKIP] 眼底領域を取得できません: "
                f"{filename}"
            )

            continue

        statistics[
            "image"
        ] = filename

        statistics[
            "mask_touches_border"
        ] = (
            preprocessor.mask_touches_border(
                mask
            )
        )

        records.append(
            statistics
        )

    if not records:

        raise RuntimeError(
            "色統計を計算できる画像がありません。"
        )

    dataframe = pd.DataFrame(
        records
    )

    return dataframe


# ============================================================
# 出力フォルダを空にする
# ============================================================


def clear_output_directory(
    directory
):

    os.makedirs(
        directory,
        exist_ok=True
    )

    for name in os.listdir(
        directory
    ):

        path = os.path.join(
            directory,
            name
        )

        if (
            os.path.isdir(path)
            and not os.path.islink(path)
        ):

            shutil.rmtree(
                path
            )

        else:

            os.remove(
                path
            )


# ============================================================
# メイン処理
# ============================================================


def main():

    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    if not os.path.isdir(
        SOURCE_IMAGE_DIR
    ):

        raise FileNotFoundError(
            f"入力フォルダがありません: "
            f"{SOURCE_IMAGE_DIR}"
        )

    files = sorted(
        filename
        for filename in os.listdir(
            SOURCE_IMAGE_DIR
        )
        if (
            os.path.isfile(
                os.path.join(
                    SOURCE_IMAGE_DIR,
                    filename
                )
            )
            and
            os.path.splitext(
                filename
            )[1].lower()
            in SUPPORTED_EXTENSIONS
        )
    )

    if not files:

        raise FileNotFoundError(
            f"{SOURCE_IMAGE_DIR} に対応画像がありません。"
        )

    # --------------------------------------------------------
    # 色統計側だけを初期化
    # --------------------------------------------------------

    clear_output_directory(
        COLOR_VISUALIZATION_DIR
    )

    for csv_path in (
        COLOR_STATISTICS_CSV,
        COLOR_OUTLIERS_CSV
    ):

        if os.path.exists(
            csv_path
        ):

            os.remove(
                csv_path
            )

    preprocessor = FundusPreprocessor(
        image_dir=SOURCE_IMAGE_DIR,
        output_color_dir=None,
        output_green_dir=None,
        output_mask_dir=None,
        debug_dir=None,
        debug_sample_count=(
            DEBUG_SAMPLE_COUNT
        ),
        random_seed=(
            RANDOM_SEED
        ),
    )

    color_analyzer = ColorAnalyzer(
        iqr_multiplier=(
            COLOR_IQR_MULTIPLIER
        ),
        outlier_feature_count=(
            COLOR_OUTLIER_FEATURE_COUNT
        ),
    )

    # --------------------------------------------------------
    # 色統計
    # --------------------------------------------------------

    color_df = analyze_all_colors(
        files,
        SOURCE_IMAGE_DIR,
        preprocessor,
    )

    # --------------------------------------------------------
    # 外れ値判定
    #
    # mean_R
    # mean_G
    # mean_B
    # mean_L
    #
    # の4項目だけを使用
    # --------------------------------------------------------

    (
        color_df,
        color_bounds
    ) = color_analyzer.detect_outliers(
        color_df
    )

    # --------------------------------------------------------
    # 全色統計CSV
    # --------------------------------------------------------

    color_df.to_csv(
        COLOR_STATISTICS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 色外れ値CSV
    # --------------------------------------------------------

    outlier_df = color_df[
        color_df["color_outlier"]
    ].copy()

    outlier_df.to_csv(
        COLOR_OUTLIERS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 箱ひげ図
    # --------------------------------------------------------

    create_color_boxplots(
        color_df,
        color_bounds,
        COLOR_BOXPLOT_DIR,
    )

    # --------------------------------------------------------
    # RGB散布図
    # --------------------------------------------------------

    create_color_scatterplot(
        color_df,
        COLOR_VISUALIZATION_DIR,
    )

    # --------------------------------------------------------
    # 外れ値画像
    # --------------------------------------------------------

    save_outlier_previews(
        color_df,
        SOURCE_IMAGE_DIR,
        COLOR_OUTLIER_PREVIEW_DIR,
    )

    # --------------------------------------------------------
    # 結果表示
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("色統計が完了しました")
    print("=" * 72)

    print(
        f"入力画像数         : "
        f"{len(files)}"
    )

    print(
        f"統計計算成功数     : "
        f"{len(color_df)}"
    )

    print(
        f"色外れ値数         : "
        f"{int(color_df['color_outlier'].sum())}"
    )

    print()
    print(
        "外れ値判定項目     : "
        "mean_R, mean_G, mean_B, mean_L"
    )

    print(
        "外れ値判定条件     : "
        "4項目中2項目以上"
    )

    print()
    print(
        f"色統計CSV          : "
        f"{COLOR_STATISTICS_CSV}"
    )

    print(
        f"色外れ値CSV        : "
        f"{COLOR_OUTLIERS_CSV}"
    )

    print(
        f"色統計の図         : "
        f"{COLOR_VISUALIZATION_DIR}/"
    )

    print()
    print(
        "この処理は色統計を確定するときに1回だけ実行してください。"
    )

    print(
        "以後の血管抽出は vessel_extraction.py で実行します。"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
