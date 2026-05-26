from __future__ import annotations

import csv
import io
import math
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs3"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
ARCHIVE_NAMES = ["archive (1).zip"]

CHANNELS = [
    "teams_count",
    "mean_win_rate",
    "mean_points_per_game",
    "mean_goals_for_per_game",
    "mean_goals_against_per_game",
    "mean_goal_difference_per_game",
]

CHANNEL_NAMES_RU = {
    "teams_count": "Количество команд",
    "mean_win_rate": "Средняя доля побед",
    "mean_points_per_game": "Средние очки за игру",
    "mean_goals_for_per_game": "Средние заброшенные шайбы за игру",
    "mean_goals_against_per_game": "Средние пропущенные шайбы за игру",
    "mean_goal_difference_per_game": "Средняя разница шайб за игру",
}


def find_archive() -> Path:
    for folder in [BASE_DIR, Path.cwd(), BASE_DIR.parent]:
        for name in ARCHIVE_NAMES:
            path = folder / name
            if path.exists():
                return path
    raise FileNotFoundError("")


def prepare_output_dirs() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    old_zip = BASE_DIR / "outputs3.zip"
    if old_zip.exists():
        old_zip.unlink()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_from_zip(archive_path: Path, filename: str) -> pd.DataFrame:
    with zipfile.ZipFile(archive_path, "r") as archive:
        csv_name = next((name for name in archive.namelist() if Path(name).name == filename), None)
        if csv_name is None:
            raise FileNotFoundError("")
        raw = archive.read(csv_name)

    text = None
    for encoding in ["utf-8-sig", "utf-8", "cp1251"]:
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            pass
    if text is None:
        raise UnicodeDecodeError("csv", raw, 0, 1, f"Не удалось определить кодировку {filename}")

    rows = list(csv.reader(io.StringIO(text)))
    header = [item.strip() for item in rows[0]]
    fixed_rows = []
    for row in rows[1:]:
        if len(row) < len(header):
            row += [""] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[: len(header)]
        fixed_rows.append(row)

    df = pd.DataFrame(fixed_rows, columns=header)
    for col in df.columns:
        values = df[col].replace("", np.nan)
        numeric = pd.to_numeric(values, errors="coerce")
        if values.notna().sum() and numeric.notna().sum() / values.notna().sum() >= 0.85:
            df[col] = numeric
        else:
            df[col] = values
    return df


def build_team_features(teams: pd.DataFrame) -> pd.DataFrame:
    df = teams.copy()
    df.columns = df.columns.str.strip()
    if "GP" not in df.columns and "G" in df.columns:
        df["GP"] = df["G"]
    required = ["year", "tmID", "GP", "W", "L", "Pts", "GF", "GA"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"В Teams.csv отсутствуют обязательные столбцы: {missing}")

    df = df[df["GP"] > 0].copy()
    df["win_rate"] = df["W"] / df["GP"]
    df["points_per_game"] = df["Pts"] / df["GP"]
    df["goals_for_per_game"] = df["GF"] / df["GP"]
    df["goals_against_per_game"] = df["GA"] / df["GP"]
    df["goal_difference_per_game"] = (df["GF"] - df["GA"]) / df["GP"]
    return df


def build_yearly_time_series(team_features: pd.DataFrame) -> pd.DataFrame:
    grouped = team_features.groupby("year").agg(
        teams_count=("tmID", "count"),
        mean_win_rate=("win_rate", "mean"),
        mean_points_per_game=("points_per_game", "mean"),
        mean_goals_for_per_game=("goals_for_per_game", "mean"),
        mean_goals_against_per_game=("goals_against_per_game", "mean"),
        mean_goal_difference_per_game=("goal_difference_per_game", "mean"),
    )
    full_years = range(int(grouped.index.min()), int(grouped.index.max()) + 1)
    ts = grouped.reindex(full_years)
    ts.index.name = "year"
    ts["date"] = pd.to_datetime(ts.index.astype(str) + "-01-01")
    ts = ts.set_index("date")
    ts["year"] = ts.index.year
    return ts[["year"] + CHANNELS]


def descriptive_statistics(ts: pd.DataFrame) -> pd.DataFrame:
    stats = ts[CHANNELS].describe().T
    stats = stats.rename(columns={"25%": "Q1", "50%": "median", "75%": "Q3"})
    stats.insert(0, "channel", stats.index)
    stats.insert(1, "channel_ru", [CHANNEL_NAMES_RU[col] for col in stats["channel"]])
    return stats.reset_index(drop=True).round(4)


def missing_and_outliers(ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for channel in CHANNELS:
        series = ts[channel]
        mean = series.mean()
        std = series.std()
        outlier_mask = (series - mean).abs() > 3 * std if std and not math.isnan(std) else series == np.inf
        outlier_years = [str(int(year)) for year in ts.loc[outlier_mask.fillna(False), "year"]]
        rows.append(
            {
                "channel": channel,
                "channel_ru": CHANNEL_NAMES_RU[channel],
                "missing_count": int(series.isna().sum()),
                "missing_percent": round(series.isna().mean() * 100, 2),
                "outliers_3sigma_count": int(outlier_mask.sum()),
                "outlier_years": ", ".join(outlier_years) if outlier_years else "нет",
            }
        )
    return pd.DataFrame(rows)


def correlation_tables(ts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr = ts[CHANNELS].interpolate().corr().round(3)
    rows = []
    for i, first in enumerate(CHANNELS):
        for second in CHANNELS[i + 1 :]:
            rows.append(
                {
                    "first_channel": first,
                    "second_channel": second,
                    "first_channel_ru": CHANNEL_NAMES_RU[first],
                    "second_channel_ru": CHANNEL_NAMES_RU[second],
                    "pearson_r": corr.loc[first, second],
                    "abs_r": abs(corr.loc[first, second]),
                }
            )
    pairs = pd.DataFrame(rows).sort_values("abs_r", ascending=False).drop(columns="abs_r")
    return corr, pairs


def decompose_key_channel(ts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = "mean_goals_for_per_game"
    period = 10
    series = ts[key].interpolate(limit_direction="both")

    trend = (
        series.rolling(window=period, center=True, min_periods=max(3, period // 2))
        .mean()
        .interpolate(limit_direction="both")
    )
    detrended = series - trend
    seasonal_pattern = detrended.groupby(np.arange(len(detrended)) % period).mean()
    seasonal_pattern = seasonal_pattern - seasonal_pattern.mean()
    seasonal = pd.Series(
        [seasonal_pattern.iloc[i % period] for i in range(len(series))],
        index=series.index,
        name="seasonal",
    )
    residual = series - trend - seasonal
    signal = trend + seasonal

    signal_variance = float(np.nanvar(signal))
    noise_variance = float(np.nanvar(residual))
    snr = 10 * np.log10(signal_variance / noise_variance) if noise_variance else np.inf

    components = pd.DataFrame(
        {
            "year": ts["year"],
            "observed": series,
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
            "signal": signal,
        },
        index=ts.index,
    )
    snr_summary = pd.DataFrame(
        [
            {
                "key_channel": key,
                "key_channel_ru": CHANNEL_NAMES_RU[key],
                "model": "additive_moving_average",
                "period_years": period,
                "signal_variance": round(signal_variance, 6),
                "noise_variance": round(noise_variance, 6),
                "snr_db": round(float(snr), 2),
                "snr_quality": "хорошо" if snr >= 10 else "удовлетворительно" if snr >= 0 else "плохо",
            }
        ]
    )
    return components, snr_summary


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def figure_time_series(ts: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(10, 12), sharex=True)
    for ax, channel in zip(axes, CHANNELS):
        ax.plot(ts["year"], ts[channel], linewidth=1.4)
        ax.axvline(2004, linestyle="--", linewidth=1)
        ax.set_title(CHANNEL_NAMES_RU[channel], loc="left", fontsize=10)
        ax.set_ylabel("значение", fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Год сезона")
    fig.suptitle("Исходные каналы многомерного временного ряда по сезонам")
    fig.subplots_adjust(hspace=0.45)
    save_plot(FIG_DIR / "22_time_series_overview.png")


def figure_outliers_boxplots(ts: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, channel in zip(axes.ravel(), CHANNELS):
        ax.boxplot(ts[channel].dropna(), vert=True)
        ax.set_title(CHANNEL_NAMES_RU[channel], fontsize=10)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Диаграммы размаха для выявления потенциальных выбросов")
    save_plot(FIG_DIR / "23_outliers_boxplots.png")


def figure_ranges(ts: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.boxplot([ts[channel].dropna() for channel in CHANNELS])
    ax.set_xticks(range(1, len(CHANNELS) + 1))
    ax.set_xticklabels([CHANNEL_NAMES_RU[c] for c in CHANNELS])
    ax.set_title("Сравнение диапазонов значений каналов")
    ax.set_ylabel("Значение канала")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    save_plot(FIG_DIR / "24_value_ranges.png")


def figure_correlation(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr.values, vmin=-1, vmax=1)
    labels = [CHANNEL_NAMES_RU[c] for c in corr.columns]
    ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Матрица корреляции Пирсона между каналами")
    save_plot(FIG_DIR / "25_correlation_heatmap.png")


def figure_noise(components: pd.DataFrame) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(10, 12))
    x = components["year"]
    axes[0].plot(x, components["observed"], linewidth=1.4)
    axes[0].set_title("Исходный ряд: средние заброшенные шайбы за игру")
    axes[1].plot(x, components["trend"], linewidth=1.4)
    axes[1].set_title("Тренд")
    axes[2].plot(x, components["seasonal"], linewidth=1.4)
    axes[2].set_title("Сезонная компонента, период 10 лет")
    axes[3].plot(x, components["residual"], linewidth=1.2)
    axes[3].set_title("Остатки, шум")
    axes[4].hist(components["residual"].dropna(), bins=20)
    axes[4].set_title("Распределение остатков")
    axes[4].set_xlabel("Значение остатка")
    for ax in axes[:4]:
        ax.set_xlabel("Год сезона")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Декомпозиция ключевого канала и анализ шумовой компоненты")
    save_plot(FIG_DIR / "26_noise_decomposition.png")


def save_csv_excel(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    df.to_csv(path, index=index, sep=";", decimal=",", encoding="utf-8-sig")


def save_tables(archive_path: Path, teams: pd.DataFrame, ts: pd.DataFrame, stats: pd.DataFrame, missing: pd.DataFrame, corr: pd.DataFrame, pairs: pd.DataFrame, components: pd.DataFrame, snr_summary: pd.DataFrame) -> None:
    dataset_summary = pd.DataFrame(
        [
            ["Источник", "Professional Hockey Database / локальный архив " + archive_path.name],
            ["Базовая таблица", "Teams.csv"],
            ["Объект исходной строки", "Команда в конкретном сезоне"],
            ["Строк после фильтрации GP > 0", len(teams)],
            ["Временная метка", "year, преобразована в дату 1 января соответствующего года"],
            ["Период наблюдений", f"{int(ts['year'].min())}-{int(ts['year'].max())}"],
            ["Длина полной годовой сетки", len(ts)],
            ["Фактических годовых наблюдений", int(ts[CHANNELS].notna().all(axis=1).sum())],
            ["Число каналов", len(CHANNELS)],
            ["Частота дискретизации", "1 год"],
            ["Пропущенный год", "2004"],
            ["Тип задачи", "первичный анализ многомерного временного ряда; подготовка к прогнозированию"],
        ],
        columns=["parameter", "value"],
    )
    channel_description = pd.DataFrame(
        [[channel, CHANNEL_NAMES_RU[channel]] for channel in CHANNELS],
        columns=["channel", "description"],
    )
    save_csv_excel(ts, OUT_DIR / "time_series.csv")
    save_csv_excel(dataset_summary, TABLE_DIR / "dataset_summary.csv")
    save_csv_excel(channel_description, TABLE_DIR / "channel_description.csv")
    save_csv_excel(stats, TABLE_DIR / "descriptive_statistics.csv")
    save_csv_excel(missing, TABLE_DIR / "missing_outliers.csv")
    save_csv_excel(corr, TABLE_DIR / "correlation_matrix.csv", index=True)
    save_csv_excel(pairs, TABLE_DIR / "correlation_pairs.csv")
    save_csv_excel(components, TABLE_DIR / "decomposition_components.csv")
    save_csv_excel(snr_summary, TABLE_DIR / "snr_summary.csv")


def build_chapter3_outputs() -> None:
    archive_path = find_archive()
    prepare_output_dirs()
    teams_raw = read_csv_from_zip(archive_path, "Teams.csv")
    teams = build_team_features(teams_raw)
    ts = build_yearly_time_series(teams)
    stats = descriptive_statistics(ts)
    missing = missing_and_outliers(ts)
    corr, pairs = correlation_tables(ts)
    components, snr_summary = decompose_key_channel(ts)

    figure_time_series(ts)
    figure_outliers_boxplots(ts)
    figure_ranges(ts)
    figure_correlation(corr)
    figure_noise(components)
    save_tables(archive_path, teams, ts, stats, missing, corr, pairs, components, snr_summary)



if __name__ == "__main__":
    build_chapter3_outputs()
