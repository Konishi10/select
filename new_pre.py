import os
import random

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

try:
    from skimage.filters import frangi
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


# ============================================================
# 基本設定
# ============================================================

SOURCE_IMAGE_DIR = "train_images/train_images"
LABEL_CSV = "train_1.csv"

# ------------------------------------------------------------
# 前処理出力
# ------------------------------------------------------------

PREPROCESSED_COLOR_DIR = "preprocessed_images"
PREPROCESSED_GREEN_DIR = "preprocessed_green_images"
FUNDUS_MASK_DIR = "fundus_masks"
PREPROCESS_DEBUG_DIR = "preprocessing_debug_samples"

# ------------------------------------------------------------
# 色統計・外れ値検出
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

# ------------------------------------------------------------
# 色外れ値判定設定
# ------------------------------------------------------------

# IQRの倍率
# 1.5 = 一般的な箱ひげ図の外れ値基準
COLOR_IQR_MULTIPLIER = 1.5

# 何個以上の色特徴が外れ値なら除外するか
#
# 1個だけだと誤判定しやすいため、
# デフォルトでは2個以上にしている。
COLOR_OUTLIER_FEATURE_COUNT = 2

# ------------------------------------------------------------
# 血管抽出出力
# ------------------------------------------------------------

VESSEL_MASK_DIR = "vessel_masks"
SKELETON_DIR = "vessel_skeletons"
OVERLAY_DIR = "vessel_overlays"
EXTRACTION_DEBUG_DIR = "vessel_extraction_debug"

METRICS_CSV = "vessel_extraction_metrics.csv"

# ------------------------------------------------------------
# Debug
# ------------------------------------------------------------

DEBUG_SAMPLE_COUNT = 5
RANDOM_SEED = 42


# ============================================================
# 境界セーフなモルフォロジー処理
# ============================================================

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


# ============================================================
# 共通ユーティリティ
# ============================================================

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


# ============================================================
# 前処理クラス
# ============================================================

class FundusPreprocessor:

    def __init__(
        self,
        image_dir=SOURCE_IMAGE_DIR,
        output_color_dir=PREPROCESSED_COLOR_DIR,
        output_green_dir=PREPROCESSED_GREEN_DIR,
        output_mask_dir=FUNDUS_MASK_DIR,
        debug_dir=PREPROCESS_DEBUG_DIR,
        debug_sample_count=5,
        random_seed=42,
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

        os.makedirs(
            self.output_color_dir,
            exist_ok=True
        )

        os.makedirs(
            self.output_green_dir,
            exist_ok=True
        )

        os.makedirs(
            self.output_mask_dir,
            exist_ok=True
        )

        os.makedirs(
            self.debug_dir,
            exist_ok=True
        )

    # --------------------------------------------------------
    # 眼底マスク
    # --------------------------------------------------------

    def create_fundus_mask(
        self,
        img
    ):

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        _, mask = cv2.threshold(
            gray,
            10,
            255,
            cv2.THRESH_BINARY
        )

        kernel = np.ones(
            (21, 21),
            np.uint8
        )

        mask = safe_close(
            mask,
            kernel
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        clean_mask = np.zeros_like(
            mask
        )

        if len(contours) > 0:

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

        clean_mask = safe_erode(
            clean_mask,
            np.ones(
                (7, 7),
                np.uint8
            ),
            iterations=1
        )

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


# ============================================================
# 色統計クラス
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


# ============================================================
# 色統計の箱ひげ図
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


# ============================================================
# 色統計の散布図
# ============================================================

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
# 色統計処理
# ============================================================

def analyze_all_colors(
    files,
    image_dir,
    preprocessor
):

    records = []

    print()
    print("=" * 70)
    print("色統計を計算しています")
    print("=" * 70)

    for file in tqdm(
        files,
        desc="Color statistics"
    ):

        image_path = os.path.join(
            image_dir,
            file
        )

        image = cv2.imread(
            image_path
        )

        if image is None:

            print(
                f"画像を読み込めませんでした: "
                f"{image_path}"
            )

            continue

        mask = (
            preprocessor.create_fundus_mask(
                image
            )
        )

        statistics = (
            ColorAnalyzer().calculate_color_statistics(
                image,
                mask
            )
        )

        if statistics is None:

            continue

        statistics[
            "image"
        ] = file

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

    df = pd.DataFrame(
        records
    )

    return df


# ============================================================
# 血管抽出クラス
# ============================================================

class VesselExtractor:

    def __init__(self):

        # ----------------------------------------------------
        # Frangi
        # ----------------------------------------------------

        self.frangi_sigmas = [
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            4.0,
        ]

        # ----------------------------------------------------
        # 閾値
        # ----------------------------------------------------

        self.strong_frangi_percentile = 88.0
        self.candidate_percentile = 72.0

        self.strong_blackhat_percentile = 90.0
        self.candidate_blackhat_percentile = 72.0

        self.strong_darkline_percentile = 90.0
        self.candidate_darkline_percentile = 75.0

        # ----------------------------------------------------
        # seed growth
        # ----------------------------------------------------

        self.growth_iterations = 18

        # ----------------------------------------------------
        # 小ノイズ除去
        # ----------------------------------------------------

        self.min_component_area = 18

        # ----------------------------------------------------
        # skeleton
        # ----------------------------------------------------

        self.min_skeleton_length = 12

    # --------------------------------------------------------
    # Frangi
    # --------------------------------------------------------

    def build_frangi_response(
        self,
        green,
        mask
    ):

        normalized = normalize_01(
            green,
            mask
        )

        if SKIMAGE_AVAILABLE:

            response = frangi(
                normalized,
                sigmas=self.frangi_sigmas,
                black_ridges=True
            )

            response = np.nan_to_num(
                response,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            response[
                mask == 0
            ] = 0.0

            return response.astype(
                np.float32
            )

        response = np.zeros_like(
            normalized,
            dtype=np.float32
        )

        for sigma in (
            self.frangi_sigmas
        ):

            smoothed = cv2.GaussianBlur(
                normalized,
                (0, 0),
                sigmaX=sigma
            )

            dxx = cv2.Sobel(
                smoothed,
                cv2.CV_32F,
                2,
                0,
                ksize=3
            )

            dyy = cv2.Sobel(
                smoothed,
                cv2.CV_32F,
                0,
                2,
                ksize=3
            )

            dxy = cv2.Sobel(
                smoothed,
                cv2.CV_32F,
                1,
                1,
                ksize=3
            )

            tmp = np.sqrt(
                (dxx - dyy) ** 2
                + 4.0 * dxy ** 2
            )

            lambda1 = 0.5 * (
                dxx
                + dyy
                + tmp
            )

            lambda2 = 0.5 * (
                dxx
                + dyy
                - tmp
            )

            vesselness = np.maximum(
                -lambda2,
                0
            )

            vesselness = (
                vesselness
                / (
                    np.max(
                        vesselness
                    )
                    + 1e-6
                )
            )

            response = np.maximum(
                response,
                vesselness
            )

        response[
            mask == 0
        ] = 0.0

        return response.astype(
            np.float32
        )

    # --------------------------------------------------------
    # Black-hat
    # --------------------------------------------------------

    def build_blackhat(
        self,
        green,
        mask
    ):

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (15, 15)
        )

        blackhat = cv2.morphologyEx(
            green,
            cv2.MORPH_BLACKHAT,
            kernel
        )

        blackhat = normalize_01(
            blackhat,
            mask
        )

        blackhat[
            mask == 0
        ] = 0.0

        return blackhat

    # --------------------------------------------------------
    # Dark line
    # --------------------------------------------------------

    def build_dark_line(
        self,
        green,
        mask
    ):

        kernel_sizes = [
            7,
            11,
            15
        ]

        response = np.zeros_like(
            green,
            dtype=np.float32
        )

        for k in kernel_sizes:

            kernel = (
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (k, k)
                )
            )

            blackhat = cv2.morphologyEx(
                green,
                cv2.MORPH_BLACKHAT,
                kernel
            )

            blackhat = (
                blackhat.astype(
                    np.float32
                )
                / 255.0
            )

            response = np.maximum(
                response,
                blackhat
            )

        response = normalize_01(
            response,
            mask
        )

        response[
            mask == 0
        ] = 0.0

        return response

    # --------------------------------------------------------
    # Coherence
    # --------------------------------------------------------

    def build_coherence(
        self,
        green,
        mask
    ):

        img = (
            green.astype(
                np.float32
            )
            / 255.0
        )

        gx = cv2.Sobel(
            img,
            cv2.CV_32F,
            1,
            0,
            ksize=3
        )

        gy = cv2.Sobel(
            img,
            cv2.CV_32F,
            0,
            1,
            ksize=3
        )

        jxx = cv2.GaussianBlur(
            gx * gx,
            (0, 0),
            2.0
        )

        jyy = cv2.GaussianBlur(
            gy * gy,
            (0, 0),
            2.0
        )

        jxy = cv2.GaussianBlur(
            gx * gy,
            (0, 0),
            2.0
        )

        tmp = np.sqrt(
            (jxx - jyy) ** 2
            + 4.0 * jxy ** 2
        )

        coherence = (
            tmp
            / (
                jxx
                + jyy
                + 1e-6
            )
        )

        coherence = np.clip(
            coherence,
            0,
            1
        )

        coherence[
            mask == 0
        ] = 0

        return coherence.astype(
            np.float32
        )

    # --------------------------------------------------------
    # 中心
    # --------------------------------------------------------

    def estimate_center(
        self,
        mask
    ):

        ys, xs = np.where(
            mask > 0
        )

        if len(xs) == 0:

            h, w = mask.shape

            return (
                w / 2.0,
                h / 2.0
            )

        return (
            float(np.mean(xs)),
            float(np.mean(ys))
        )

    # --------------------------------------------------------
    # radial penalty
    # --------------------------------------------------------

    def build_radial_penalty(
        self,
        shape,
        mask
    ):

        h, w = shape

        cx, cy = (
            self.estimate_center(
                mask
            )
        )

        yy, xx = np.indices(
            (h, w),
            dtype=np.float32
        )

        radius = np.sqrt(
            (xx - cx) ** 2
            + (yy - cy) ** 2
        )

        radius_norm = (
            radius
            / (
                np.max(
                    radius[
                        mask > 0
                    ]
                )
                + 1e-6
            )
        )

        radial_blur = cv2.GaussianBlur(
            radius_norm,
            (0, 0),
            5.0
        )

        penalty = np.abs(
            radius_norm
            - radial_blur
        )

        penalty = normalize_01(
            penalty,
            mask
        )

        penalty[
            mask == 0
        ] = 1.0

        return penalty

    # --------------------------------------------------------
    # ring penalty
    # --------------------------------------------------------

    def build_ring_penalty(
        self,
        response,
        mask
    ):

        h, w = response.shape

        cx, cy = (
            self.estimate_center(
                mask
            )
        )

        yy, xx = np.indices(
            (h, w),
            dtype=np.float32
        )

        radius = np.sqrt(
            (xx - cx) ** 2
            + (yy - cy) ** 2
        )

        max_radius = (
            np.max(
                radius[
                    mask > 0
                ]
            )
            + 1e-6
        )

        ring_score = np.zeros_like(
            response,
            dtype=np.float32
        )

        for sigma in [
            8.0,
            15.0,
            25.0
        ]:

            blurred = cv2.GaussianBlur(
                response,
                (0, 0),
                sigma
            )

            ring_score = np.maximum(
                ring_score,
                blurred
            )

        ring_score = (
            ring_score
            / (
                np.max(
                    ring_score[
                        mask > 0
                    ]
                )
                + 1e-6
            )
        )

        outer_weight = np.clip(
            radius
            / max_radius,
            0,
            1
        )

        penalty = (
            0.75 * ring_score
            + 0.25 * outer_weight
        )

        penalty = np.clip(
            penalty,
            0,
            1
        )

        penalty[
            mask == 0
        ] = 1.0

        return penalty.astype(
            np.float32
        )

    # --------------------------------------------------------
    # local relative
    # --------------------------------------------------------

    def build_local_relative(
        self,
        green,
        mask
    ):

        green_float = (
            green.astype(
                np.float32
            )
            / 255.0
        )

        local_background = (
            cv2.GaussianBlur(
                green_float,
                (0, 0),
                7.0
            )
        )

        relative = (
            local_background
            - green_float
        )

        relative = np.maximum(
            relative,
            0
        )

        relative = normalize_01(
            relative,
            mask
        )

        relative[
            mask == 0
        ] = 0

        return relative

    # --------------------------------------------------------
    # strong seed
    # --------------------------------------------------------

    def build_strong_seed(
        self,
        frangi_response,
        blackhat,
        dark_line,
        coherence,
        mask
    ):

        frangi_seed = (
            threshold_percentile(
                frangi_response,
                mask,
                self.strong_frangi_percentile
            )
        )

        blackhat_seed = (
            threshold_percentile(
                blackhat,
                mask,
                self.strong_blackhat_percentile
            )
        )

        darkline_seed = (
            threshold_percentile(
                dark_line,
                mask,
                self.strong_darkline_percentile
            )
        )

        coherence_seed = (
            coherence > 0.30
        ).astype(
            np.uint8
        ) * 255

        score = (

            (
                frangi_seed > 0
            ).astype(np.uint8)

            +

            (
                blackhat_seed > 0
            ).astype(np.uint8)

            +

            (
                darkline_seed > 0
            ).astype(np.uint8)

            +

            (
                coherence_seed > 0
            ).astype(np.uint8)

        )

        seed = (
            (
                score >= 2
            )
            & (
                mask > 0
            )
        ).astype(
            np.uint8
        ) * 255

        seed = (
            remove_small_components(
                seed,
                8
            )
        )

        return seed

    # --------------------------------------------------------
    # candidate
    # --------------------------------------------------------

    def build_candidate(
        self,
        frangi_response,
        blackhat,
        dark_line,
        local_relative,
        coherence,
        radial_penalty,
        ring_penalty,
        mask
    ):

        score = (

            0.40
            * frangi_response

            +

            0.25
            * blackhat

            +

            0.20
            * dark_line

            +

            0.15
            * local_relative
        )

        score *= (
            0.45
            + 0.55 * coherence
        )

        score *= (
            1.0
            - 0.45
            * radial_penalty
        )

        score *= (
            1.0
            - 0.35
            * ring_penalty
        )

        score[
            mask == 0
        ] = 0

        candidate = (
            threshold_percentile(
                score,
                mask,
                self.candidate_percentile
            )
        )

        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_OPEN,
            np.ones(
                (2, 2),
                np.uint8
            )
        )

        return (
            candidate,
            score
        )

    # --------------------------------------------------------
    # seed growth
    # --------------------------------------------------------

    def seed_growth(
        self,
        seed,
        candidate,
        score,
        mask
    ):

        result = seed.copy()

        candidate_bool = (
            candidate > 0
        )

        score_threshold = np.percentile(
            score[
                mask > 0
            ],
            55.0
        )

        for _ in range(
            self.growth_iterations
        ):

            previous = result.copy()

            dilated = safe_dilate(
                result,
                np.ones(
                    (3, 3),
                    np.uint8
                ),
                iterations=1
            )

            neighbor = (
                dilated > 0
            )

            add = (
                neighbor
                & candidate_bool
                & (mask > 0)
            )

            add &= (
                score
                >= score_threshold
            )

            result[
                add
            ] = 255

            if np.array_equal(
                previous,
                result
            ):

                break

        return result

    # --------------------------------------------------------
    # skeleton短枝除去
    # --------------------------------------------------------

    def remove_short_skeleton_components(
        self,
        skeleton
    ):

        binary = (
            skeleton > 0
        ).astype(
            np.uint8
        )

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8
            )
        )

        result = np.zeros_like(
            skeleton
        )

        for label in range(
            1,
            num_labels
        ):

            component = (
                labels == label
            )

            length = np.count_nonzero(
                component
            )

            if (
                length
                >= self.min_skeleton_length
            ):

                result[
                    component
                ] = 255

        return result

    # --------------------------------------------------------
    # skeleton
    # --------------------------------------------------------

    def skeletonize(
        self,
        binary
    ):

        if SKIMAGE_AVAILABLE:

            from skimage.morphology import (
                skeletonize
            )

            skel = skeletonize(
                binary > 0
            )

            return (
                skel.astype(
                    np.uint8
                ) * 255
            )

        img = binary.copy()

        skel = np.zeros_like(
            img
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            (3, 3)
        )

        while True:

            eroded = cv2.erode(
                img,
                kernel
            )

            temp = cv2.dilate(
                eroded,
                kernel
            )

            temp = cv2.subtract(
                img,
                temp
            )

            skel = cv2.bitwise_or(
                skel,
                temp
            )

            img = eroded.copy()

            if cv2.countNonZero(
                img
            ) == 0:

                break

        return skel

    # --------------------------------------------------------
    # overlay
    # --------------------------------------------------------

    def make_overlay(
        self,
        original,
        skeleton
    ):

        overlay = original.copy()

        overlay[
            skeleton > 0
        ] = (
            0,
            0,
            255
        )

        return overlay

    # --------------------------------------------------------
    # 1枚の血管抽出
    # --------------------------------------------------------

    def extract(
        self,
        original,
        green,
        fundus_mask
    ):

        steps = {}

        steps[
            "01_fundus_mask"
        ] = fundus_mask.copy()

        # ----------------------------------------------------
        # Frangi
        # ----------------------------------------------------

        frangi_response = (
            self.build_frangi_response(
                green,
                fundus_mask
            )
        )

        steps[
            "02_frangi"
        ] = to_uint8(
            normalize_01(
                frangi_response,
                fundus_mask
            )
        )

        # ----------------------------------------------------
        # local relative
        # ----------------------------------------------------

        local_relative = (
            self.build_local_relative(
                green,
                fundus_mask
            )
        )

        steps[
            "03_local_relative"
        ] = to_uint8(
            local_relative
        )

        # ----------------------------------------------------
        # Black-hat
        # ----------------------------------------------------

        blackhat = (
            self.build_blackhat(
                green,
                fundus_mask
            )
        )

        steps[
            "04_blackhat"
        ] = to_uint8(
            blackhat
        )

        # ----------------------------------------------------
        # dark-line
        # ----------------------------------------------------

        dark_line = (
            self.build_dark_line(
                green,
                fundus_mask
            )
        )

        steps[
            "05_dark_line"
        ] = to_uint8(
            dark_line
        )

        # ----------------------------------------------------
        # coherence
        # ----------------------------------------------------

        coherence = (
            self.build_coherence(
                green,
                fundus_mask
            )
        )

        steps[
            "06_coherence"
        ] = to_uint8(
            coherence
        )

        # ----------------------------------------------------
        # radial penalty
        # ----------------------------------------------------

        radial_penalty = (
            self.build_radial_penalty(
                green.shape,
                fundus_mask
            )
        )

        steps[
            "07_radial_penalty"
        ] = to_uint8(
            radial_penalty
        )

        # ----------------------------------------------------
        # ring penalty
        # ----------------------------------------------------

        ring_penalty = (
            self.build_ring_penalty(
                frangi_response,
                fundus_mask
            )
        )

        steps[
            "08_ring_penalty"
        ] = to_uint8(
            ring_penalty
        )

        # ----------------------------------------------------
        # strong seed
        # ----------------------------------------------------

        strong_seed = (
            self.build_strong_seed(
                frangi_response,
                blackhat,
                dark_line,
                coherence,
                fundus_mask
            )
        )

        steps[
            "09_strong_seed"
        ] = strong_seed.copy()

        # ----------------------------------------------------
        # candidate
        # ----------------------------------------------------

        candidate, score = (
            self.build_candidate(
                frangi_response,
                blackhat,
                dark_line,
                local_relative,
                coherence,
                radial_penalty,
                ring_penalty,
                fundus_mask
            )
        )

        steps[
            "10_candidate"
        ] = candidate.copy()

        # ----------------------------------------------------
        # seed growth
        # ----------------------------------------------------

        seed_growth = (
            self.seed_growth(
                strong_seed,
                candidate,
                score,
                fundus_mask
            )
        )

        steps[
            "11_seed_growth"
        ] = seed_growth.copy()

        # ----------------------------------------------------
        # close
        # ----------------------------------------------------

        vessel_mask = safe_close(
            seed_growth,
            np.ones(
                (3, 3),
                np.uint8
            )
        )

        vessel_mask[
            fundus_mask == 0
        ] = 0

        # ----------------------------------------------------
        # 小領域除去
        # ----------------------------------------------------

        vessel_mask = (
            remove_small_components(
                vessel_mask,
                self.min_component_area
            )
        )

        vessel_mask[
            fundus_mask == 0
        ] = 0

        steps[
            "12_vessel_mask"
        ] = vessel_mask.copy()

        # ----------------------------------------------------
        # skeleton
        # ----------------------------------------------------

        skeleton = (
            self.skeletonize(
                vessel_mask
            )
        )

        skeleton = (
            self.remove_short_skeleton_components(
                skeleton
            )
        )

        steps[
            "13_skeleton"
        ] = skeleton.copy()

        # ----------------------------------------------------
        # overlay
        # ----------------------------------------------------

        overlay = (
            self.make_overlay(
                original,
                skeleton
            )
        )

        steps[
            "14_overlay"
        ] = overlay.copy()

        return (
            vessel_mask,
            skeleton,
            overlay,
            steps
        )


# ============================================================
# デバッグ画像保存
# ============================================================

def save_extraction_debug(
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
# 出力フォルダを空にする
# ============================================================

def clear_output_directory(
    directory
):

    os.makedirs(
        directory,
        exist_ok=True
    )

    for file in os.listdir(
        directory
    ):

        path = os.path.join(
            directory,
            file
        )

        if os.path.isfile(
            path
        ):

            try:

                os.remove(
                    path
                )

            except Exception:

                pass

        elif os.path.isdir(
            path
        ):

            import shutil

            try:

                shutil.rmtree(
                    path
                )

            except Exception:

                pass


# ============================================================
# メイン
# ============================================================

def main():

    # ========================================================
    # 出力フォルダ
    # ========================================================

    output_dirs = [

        PREPROCESSED_COLOR_DIR,

        PREPROCESSED_GREEN_DIR,

        FUNDUS_MASK_DIR,

        PREPROCESS_DEBUG_DIR,

        VESSEL_MASK_DIR,

        SKELETON_DIR,

        OVERLAY_DIR,

        EXTRACTION_DEBUG_DIR,

        COLOR_VISUALIZATION_DIR,

        COLOR_BOXPLOT_DIR,

        COLOR_OUTLIER_PREVIEW_DIR,

    ]

    for directory in output_dirs:

        clear_output_directory(
            directory
        )

    # ========================================================
    # 前処理
    # ========================================================

    preprocessor = (
        FundusPreprocessor(
            image_dir=SOURCE_IMAGE_DIR,
            output_color_dir=PREPROCESSED_COLOR_DIR,
            output_green_dir=PREPROCESSED_GREEN_DIR,
            output_mask_dir=FUNDUS_MASK_DIR,
            debug_dir=PREPROCESS_DEBUG_DIR,
            debug_sample_count=DEBUG_SAMPLE_COUNT,
            random_seed=RANDOM_SEED,
        )
    )

    # ========================================================
    # 血管抽出
    # ========================================================

    extractor = (
        VesselExtractor()
    )

    # ========================================================
    # 色解析
    # ========================================================

    color_analyzer = (
        ColorAnalyzer(
            iqr_multiplier=COLOR_IQR_MULTIPLIER,
            outlier_feature_count=COLOR_OUTLIER_FEATURE_COUNT
        )
    )

    # ========================================================
    # 入力データ
    # ========================================================
    # train_1.csvからdiagnosis=0（正常）の画像IDだけを取得し、
    # 元画像フォルダから直接読み込む。
    #
    # healthy_imagesフォルダへのコピーは行わない。
    # ========================================================

    if not os.path.isfile(LABEL_CSV):

        raise FileNotFoundError(
            f"ラベルCSVがありません: {LABEL_CSV}"
        )

    if not os.path.isdir(SOURCE_IMAGE_DIR):

        raise FileNotFoundError(
            f"元画像フォルダがありません: {SOURCE_IMAGE_DIR}"
        )

    label_df = pd.read_csv(LABEL_CSV)

    required_columns = {"id_code", "diagnosis"}

    missing_columns = required_columns - set(label_df.columns)

    if missing_columns:

        raise ValueError(
            "CSVに必要な列がありません: "
            + ", ".join(sorted(missing_columns))
        )

    healthy_df = label_df[
        pd.to_numeric(
            label_df["diagnosis"],
            errors="coerce"
        ) == 0
    ].copy()

    # CSVのid_codeから直接ファイル名を作る。
    # 中間フォルダhealthy_imagesは作成しない。
    candidate_files = [
        str(image_id) + ".png"
        for image_id in healthy_df["id_code"].tolist()
    ]

    files = []
    missing_files = []

    for file in candidate_files:

        image_path = os.path.join(
            SOURCE_IMAGE_DIR,
            file
        )

        if os.path.isfile(image_path):
            files.append(file)
        else:
            missing_files.append(file)

    files = sorted(files)

    print()
    print("=" * 70)
    print("正常画像の選択")
    print("=" * 70)
    print(f"CSV内の正常画像数 : {len(healthy_df)}")
    print(f"実際に存在する画像 : {len(files)}")
    print(f"見つからない画像数 : {len(missing_files)}")

    if missing_files:
        for file in missing_files[:20]:
            print(f"  [MISSING] {file}")

        if len(missing_files) > 20:
            print(
                f"  ... 他 {len(missing_files) - 20} 枚"
            )

    if len(files) == 0:

        raise FileNotFoundError(
            "diagnosis=0の画像が元画像フォルダにありません。"
        )

    # ========================================================
    # ========================================================
    # STEP 0
    # 全画像の色統計
    # ========================================================
    # ========================================================

    color_df = analyze_all_colors(
        files,
        SOURCE_IMAGE_DIR,
        preprocessor
    )

    if len(color_df) == 0:

        raise RuntimeError(
            "色統計を計算できる画像がありません。"
        )

    # ========================================================
    # 外れ値検出
    # ========================================================

    (
        color_df,
        color_bounds
    ) = color_analyzer.detect_outliers(
        color_df
    )

    # ========================================================
    # CSV保存
    # ========================================================

    color_df.to_csv(
        COLOR_STATISTICS_CSV,
        index=False
    )

    outlier_df = color_df[
        color_df["color_outlier"]
    ].copy()

    outlier_df.to_csv(
        COLOR_OUTLIERS_CSV,
        index=False
    )

    # ========================================================
    # 箱ひげ図
    # ========================================================

    create_color_boxplots(
        color_df,
        color_bounds,
        COLOR_BOXPLOT_DIR
    )

    # ========================================================
    # 散布図
    # ========================================================

    create_color_scatterplot(
        color_df,
        COLOR_VISUALIZATION_DIR
    )

    # ========================================================
    # 外れ値画像の保存
    # ========================================================

    save_outlier_previews(
        color_df,
        SOURCE_IMAGE_DIR,
        COLOR_OUTLIER_PREVIEW_DIR
    )

    # ========================================================
    # 外れ値画像一覧
    # ========================================================

    excluded_files = set(
        outlier_df[
            "image"
        ].tolist()
    )

    valid_files = [

        file

        for file in files

        if file not in excluded_files

    ]

    # ========================================================
    # 色外れ値の結果表示
    # ========================================================

    print()
    print("=" * 70)
    print("色統計・外れ値検出結果")
    print("=" * 70)

    print(
        f"全画像数        : {len(files)}"
    )

    print(
        f"色統計計算数    : {len(color_df)}"
    )

    print(
        f"色外れ値画像数  : {len(excluded_files)}"
    )

    print(
        f"後段処理画像数  : {len(valid_files)}"
    )

    print(
        f"除外割合        : "
        f"{len(excluded_files) / max(len(files), 1) * 100:.2f}%"
    )

    if len(outlier_df) > 0:

        print()
        print("除外された画像:")

        for _, row in outlier_df.iterrows():

            print(
                f"  {row['image']} "
                f"-> "
                f"{row['outlier_features']}"
            )

    # ========================================================
    # Debug対象
    #
    # 色外れ値ではない画像から選ぶ
    # ========================================================

    random.seed(
        RANDOM_SEED
    )

    debug_files = set(
        random.sample(
            valid_files,
            min(
                DEBUG_SAMPLE_COUNT,
                len(valid_files)
            )
        )
    )

    records = []

    debug_index = 1

    # ========================================================
    # ========================================================
    # STEP 1以降
    # 色外れ値を除外して処理
    # ========================================================
    # ========================================================

    for file in tqdm(
        valid_files,
        desc="Preprocess + Vessel extraction"
    ):

        image_path = os.path.join(
            SOURCE_IMAGE_DIR,
            file
        )

        original = cv2.imread(
            image_path
        )

        if original is None:

            print(
                f"画像を読み込めませんでした: "
                f"{image_path}"
            )

            continue

        # ====================================================
        # STEP 1
        # 前処理
        # ====================================================

        (
            green_preprocessed,
            color_preprocessed,
            fundus_mask,
            preprocess_steps,
            before_metrics,
            after_metrics,
            diagnostics,
        ) = preprocessor.preprocess_one_image(
            original
        )

        image_id = os.path.splitext(
            file
        )[0]

        # ----------------------------------------------------
        # Green
        # ----------------------------------------------------

        green_output_path = os.path.join(
            PREPROCESSED_GREEN_DIR,
            image_id
            + "_green_preprocessed.png"
        )

        # ----------------------------------------------------
        # Color
        # ----------------------------------------------------

        color_output_path = os.path.join(
            PREPROCESSED_COLOR_DIR,
            image_id
            + "_preprocessed.png"
        )

        # ----------------------------------------------------
        # Mask
        # ----------------------------------------------------

        mask_output_path = os.path.join(
            FUNDUS_MASK_DIR,
            image_id
            + "_fundus_mask.png"
        )

        cv2.imwrite(
            green_output_path,
            green_preprocessed
        )

        cv2.imwrite(
            color_output_path,
            color_preprocessed
        )

        cv2.imwrite(
            mask_output_path,
            fundus_mask
        )

        # ====================================================
        # STEP 2
        # 血管抽出
        # ====================================================

        (
            vessel_mask,
            skeleton,
            overlay,
            extraction_steps,
        ) = extractor.extract(
            original,
            green_preprocessed,
            fundus_mask
        )

        # ----------------------------------------------------
        # Vessel mask
        # ----------------------------------------------------

        vessel_path = os.path.join(
            VESSEL_MASK_DIR,
            image_id
            + "_vessel_mask.png"
        )

        cv2.imwrite(
            vessel_path,
            vessel_mask
        )

        # ----------------------------------------------------
        # Skeleton
        # ----------------------------------------------------

        skeleton_path = os.path.join(
            SKELETON_DIR,
            image_id
            + "_skeleton.png"
        )

        cv2.imwrite(
            skeleton_path,
            skeleton
        )

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        overlay_path = os.path.join(
            OVERLAY_DIR,
            image_id
            + "_overlay.png"
        )

        cv2.imwrite(
            overlay_path,
            overlay
        )

        # ====================================================
        # Debug
        # ====================================================

        if file in debug_files:

            sample_dir = os.path.join(
                EXTRACTION_DEBUG_DIR,
                f"sample{debug_index:02d}_{image_id}"
            )

            save_extraction_debug(
                extraction_steps,
                sample_dir
            )

            preprocess_sample_dir = os.path.join(
                PREPROCESS_DEBUG_DIR,
                f"sample{debug_index:02d}_{image_id}"
            )

            preprocessor.save_debug_steps(
                preprocess_steps,
                preprocess_sample_dir
            )

            debug_index += 1

        # ====================================================
        # 色統計情報を取得
        # ====================================================

        color_row = color_df[
            color_df["image"] == file
        ]

        if len(color_row) > 0:

            color_row = (
                color_row.iloc[0]
            )

            color_outlier_feature_count = int(
                color_row[
                    "color_outlier_feature_count"
                ]
            )

            color_outlier_features = (
                color_row[
                    "outlier_features"
                ]
            )

        else:

            color_outlier_feature_count = 0

            color_outlier_features = ""

        # ====================================================
        # 血管指標
        # ====================================================

        fundus_pixels = (
            np.count_nonzero(
                fundus_mask
            )
        )

        vessel_pixels = (
            np.count_nonzero(
                vessel_mask
            )
        )

        skeleton_pixels = (
            np.count_nonzero(
                skeleton
            )
        )

        vessel_ratio = (
            vessel_pixels
            / max(
                fundus_pixels,
                1
            )
        )

        skeleton_ratio = (
            skeleton_pixels
            / max(
                fundus_pixels,
                1
            )
        )

        # ====================================================
        # CSV record
        # ====================================================

        records.append({

            "image":
                file,

            # ------------------------------------------------
            # 色外れ値
            # ------------------------------------------------

            "color_outlier":
                False,

            "color_outlier_feature_count":
                color_outlier_feature_count,

            "outlier_features":
                color_outlier_features,

            # ------------------------------------------------
            # 前処理前
            # ------------------------------------------------

            "before_mean":
                before_metrics["mean"],

            "before_std":
                before_metrics["std"],

            "before_min":
                before_metrics["min"],

            "before_max":
                before_metrics["max"],

            "before_dark_ratio":
                before_metrics["dark_ratio"],

            # ------------------------------------------------
            # 前処理後
            # ------------------------------------------------

            "after_mean":
                after_metrics["mean"],

            "after_std":
                after_metrics["std"],

            "after_min":
                after_metrics["min"],

            "after_max":
                after_metrics["max"],

            "after_dark_ratio":
                after_metrics["dark_ratio"],

            # ------------------------------------------------
            # Diagnostics
            # ------------------------------------------------

            "noise_std":
                diagnostics["noise_std"],

            "pre_denoise_h":
                diagnostics["pre_denoise_h"],

            "brightness_scale":
                diagnostics["brightness_scale"],

            "clahe_clip_limit":
                diagnostics["clahe_clip_limit"],

            "is_dark":
                diagnostics["is_dark"],

            "is_noisy":
                diagnostics["is_noisy"],

            "touches_border":
                diagnostics["touches_border"],

            # ------------------------------------------------
            # Vessel
            # ------------------------------------------------

            "fundus_pixels":
                fundus_pixels,

            "vessel_pixels":
                vessel_pixels,

            "skeleton_pixels":
                skeleton_pixels,

            "vessel_ratio":
                vessel_ratio,

            "skeleton_ratio":
                skeleton_ratio,

            # ------------------------------------------------
            # Output
            # ------------------------------------------------

            "green_preprocessed":
                green_output_path,

            "vessel_mask":
                vessel_path,

            "skeleton":
                skeleton_path,

            "overlay":
                overlay_path,
        })

    # ========================================================
    # 血管抽出CSV
    # ========================================================

    df = pd.DataFrame(
        records
    )

    df.to_csv(
        METRICS_CSV,
        index=False
    )

    # ========================================================
    # 完了
    # ========================================================

    print()
    print("=" * 70)
    print("処理が完了しました")
    print("=" * 70)

    print()
    print("[入力]")

    print(
        f"元画像フォルダ     : {SOURCE_IMAGE_DIR}"
    )

    print(
        f"ラベルCSV           : {LABEL_CSV}"
    )

    print()
    print("[色統計]")

    print(
        f"色統計CSV          : "
        f"{COLOR_STATISTICS_CSV}"
    )

    print(
        f"色外れ値CSV        : "
        f"{COLOR_OUTLIERS_CSV}"
    )

    print(
        f"箱ひげ図           : "
        f"{COLOR_BOXPLOT_DIR}"
    )

    print(
        f"色分布散布図       : "
        f"{COLOR_VISUALIZATION_DIR}"
    )

    print(
        f"外れ値画像         : "
        f"{COLOR_OUTLIER_PREVIEW_DIR}"
    )

    print()
    print("[前処理]")

    print(
        f"前処理カラー画像   : "
        f"{PREPROCESSED_COLOR_DIR}"
    )

    print(
        f"前処理Green画像     : "
        f"{PREPROCESSED_GREEN_DIR}"
    )

    print(
        f"眼底マスク          : "
        f"{FUNDUS_MASK_DIR}"
    )

    print()
    print("[血管抽出]")

    print(
        f"血管マスク          : "
        f"{VESSEL_MASK_DIR}"
    )

    print(
        f"Skeleton             : "
        f"{SKELETON_DIR}"
    )

    print(
        f"Overlay              : "
        f"{OVERLAY_DIR}"
    )

    print(
        f"前処理Debug          : "
        f"{PREPROCESS_DEBUG_DIR}"
    )

    print(
        f"血管抽出Debug        : "
        f"{EXTRACTION_DEBUG_DIR}"
    )

    print(
        f"Metrics CSV          : "
        f"{METRICS_CSV}"
    )

    print()
    print("=" * 70)
    print(
        f"全画像       : {len(files)}"
    )

    print(
        f"色外れ値     : {len(excluded_files)}"
    )

    print(
        f"採用画像     : {len(valid_files)}"
    )

    print("=" * 70)


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":

    main()