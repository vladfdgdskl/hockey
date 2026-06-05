from pathlib import Path
import csv
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_PATH = BASE_DIR / "archive (1).zip"
DATA_DIR = BASE_DIR / "hockey_data"
OUT_DIR = BASE_DIR / "analysis_outputs"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

for folder in [DATA_DIR, FIG_DIR, TABLE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def unpack_archive():
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(f"Не найден файл: {ARCHIVE_PATH}")

    with zipfile.ZipFile(ARCHIVE_PATH, "r") as archive:
        archive.extractall(DATA_DIR)


def find_file(filename):
    files = list(DATA_DIR.rglob(filename))
    if not files:
        raise FileNotFoundError(f"Не найден файл {filename} в папке {DATA_DIR}")
    return files[0]


def read_csv_fixed(filename):
    path = find_file(filename)

    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))

    header = rows[0]
    fixed_rows = []

    for row in rows[1:]:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[:len(header)]
        fixed_rows.append(row)

    df = pd.DataFrame(fixed_rows, columns=header)
    return convert_columns(df)


def convert_columns(df):
    result = df.copy()

    for column in result.columns:
        values = result[column].replace("", np.nan)
        numeric = pd.to_numeric(values, errors="coerce")

        filled_count = values.notna().sum()
        numeric_count = numeric.notna().sum()

        if filled_count > 0 and numeric_count / filled_count >= 0.85:
            result[column] = numeric
        else:
            result[column] = values

    return result


def save_table(df, name):
    df.to_csv(TABLE_DIR / name, index=False, encoding="utf-8-sig")


def save_plot(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=300)
    plt.close()


def prepare_teams(df):
    result = df.copy()

    if "GP" not in result.columns and "G" in result.columns:
        result["GP"] = result["G"]

    need = ["GP", "W", "L", "Pts", "GF", "GA"]
    missing = [column for column in need if column not in result.columns]

    if missing:
        raise ValueError(f"В Teams.csv отсутствуют необходимые колонки: {missing}")

    result = result[result["GP"] > 0].copy()

    result["win_rate"] = result["W"] / result["GP"]
    result["loss_rate"] = result["L"] / result["GP"]
    result["goals_for_per_game"] = result["GF"] / result["GP"]
    result["goals_against_per_game"] = result["GA"] / result["GP"]
    result["goal_difference"] = result["GF"] - result["GA"]
    result["goal_difference_per_game"] = result["goal_difference"] / result["GP"]

    max_for = result["goals_for_per_game"].max()
    max_against = result["goals_against_per_game"].max()

    result["recommendation_score"] = (
        0.45 * result["win_rate"]
        + 0.35 * (result["goals_for_per_game"] / max_for)
        + 0.20 * (1 - result["goals_against_per_game"] / max_against)
    )

    result["offense_level"] = pd.cut(
        result["goals_for_per_game"],
        bins=[-np.inf, 2.5, 3.5, np.inf],
        labels=["низкая", "средняя", "высокая"]
    )

    result["season_quality"] = pd.cut(
        result["win_rate"],
        bins=[-np.inf, 0.35, 0.55, np.inf],
        labels=["слабый сезон", "средний сезон", "сильный сезон"]
    )

    result["points_level"] = pd.cut(
        result["Pts"],
        bins=[-np.inf, 50, 80, np.inf],
        labels=["мало очков", "средне очков", "много очков"]
    )

    result["defense_level"] = pd.cut(
        result["goals_against_per_game"],
        bins=[-np.inf, 2.5, 3.5, np.inf],
        labels=["надежная", "средняя", "слабая"]
    )

    return result


def plot_histograms(df):
    columns = [
        "GP", "W", "L", "Pts", "GF", "GA",
        "win_rate", "goals_for_per_game",
        "goals_against_per_game", "goal_difference_per_game",
        "recommendation_score"
    ]

    columns = [column for column in columns if column in df.columns]

    df[columns].hist(bins=30, figsize=(16, 12))
    save_plot("01_histograms.png")


def plot_seaborn(df):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="win_rate", y="Pts", hue="offense_level")
    save_plot("02_win_rate_pts.png")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="goals_for_per_game",
        y="goals_against_per_game",
        hue="season_quality"
    )
    save_plot("03_goals_for_goals_against.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="offense_level", y="recommendation_score")
    save_plot("04_score_by_offense.png")


def plot_heatmaps(df):
    columns = [
        "GP", "W", "L", "Pts", "GF", "GA",
        "win_rate", "goals_for_per_game",
        "goals_against_per_game", "goal_difference_per_game",
        "recommendation_score"
    ]

    columns = [column for column in columns if column in df.columns]

    plt.figure(figsize=(12, 8))
    sns.heatmap(df[columns].corr(), annot=True, fmt=".2f")
    save_plot("05_correlation_heatmap.png")

    plt.figure(figsize=(12, 5))
    sns.heatmap(df.isna(), cbar=False)
    save_plot("06_missing_values_heatmap.png")


def plot_plotly(df):
    hover_columns = [column for column in ["year", "tmID", "name", "Pts", "GF", "GA"] if column in df.columns]

    fig = px.scatter(
        df,
        x="goals_for_per_game",
        y="win_rate",
        color="offense_level",
        hover_data=hover_columns,
        title="Результативность и процент побед"
    )
    fig.write_html(FIG_DIR / "07_plotly_goals_win_rate.html")

    fig = px.histogram(
        df,
        x="recommendation_score",
        color="season_quality",
        nbins=30,
        title="Распределение рекомендательного рейтинга"
    )
    fig.write_html(FIG_DIR / "08_plotly_recommendation_score.html")


def missing_values(df):
    table = df.isna().sum().reset_index()
    table.columns = ["feature", "missing_count"]
    table["missing_percent"] = (table["missing_count"] / len(df) * 100).round(2)
    save_table(table, "missing_values.csv")
    return table


def duplicates(df):
    count_before = len(df)
    result = df.drop_duplicates().copy()
    count_after = len(result)

    table = pd.DataFrame({
        "stage": ["before", "after"],
        "rows": [count_before, count_after]
    })

    save_table(table, "duplicates.csv")
    return result


def outliers(df):
    rows = []

    for column in [
        "win_rate",
        "goals_for_per_game",
        "goals_against_per_game",
        "goal_difference_per_game",
        "recommendation_score"
    ]:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        count = df[(df[column] < low) | (df[column] > high)].shape[0]

        rows.append({
            "feature": column,
            "outlier_count": count,
            "outlier_percent": round(count / len(df) * 100, 2)
        })

        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[column])
        save_plot(f"boxplot_{column}.png")

    table = pd.DataFrame(rows)
    save_table(table, "outliers.csv")
    return table


def filters(df):
    f1 = df[df["win_rate"] >= 0.60]
    f2 = df[df["goals_for_per_game"] >= 4.0]
    f3 = df[(df["goals_for_per_game"] >= 3.5) & (df["goal_difference_per_game"] > 0)]

    save_table(f1, "filter_win_rate_60.csv")
    save_table(f2, "filter_goals_4.csv")
    save_table(f3, "filter_attack_positive_difference.csv")

    table = pd.DataFrame({
        "filter": [
            "win_rate >= 0.60",
            "goals_for_per_game >= 4.0",
            "goals_for_per_game >= 3.5 and goal_difference_per_game > 0"
        ],
        "rows": [len(f1), len(f2), len(f3)]
    })

    save_table(table, "filters.csv")
    return f1, f2, f3


def add_noise(df):
    np.random.seed(42)
    result = df.copy()

    result["goals_for_per_game_noisy"] = (
        result["goals_for_per_game"]
        + np.random.normal(0, result["goals_for_per_game"].std() * 0.03, len(result))
    )

    result["goals_against_per_game_noisy"] = (
        result["goals_against_per_game"]
        + np.random.normal(0, result["goals_against_per_game"].std() * 0.03, len(result))
    )

    save_table(result, "teams_with_noise.csv")
    return result


def categorical_analysis(df):
    columns = ["lgID", "tmID", "offense_level", "season_quality", "points_level", "defense_level"]
    columns = [column for column in columns if column in df.columns]

    encoded = pd.get_dummies(df[columns], dummy_na=True)
    save_table(encoded, "encoded_categories.csv")

    for column in ["offense_level", "season_quality", "points_level", "defense_level"]:
        counts = df[column].value_counts(dropna=False).reset_index()
        counts.columns = [column, "count"]
        save_table(counts, f"category_{column}.csv")

        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=column)
        plt.xticks(rotation=20)
        save_plot(f"category_{column}.png")


def group_tables(teams, team_vs_team):
    if "tmID" in teams.columns:
        profile = teams.groupby("tmID").agg({
            "GP": "sum",
            "W": "sum",
            "L": "sum",
            "Pts": "sum",
            "GF": "sum",
            "GA": "sum",
            "win_rate": "mean",
            "goals_for_per_game": "mean",
            "goals_against_per_game": "mean",
            "recommendation_score": "mean"
        }).reset_index()

        profile = profile.sort_values("recommendation_score", ascending=False)
        save_table(profile, "team_profiles.csv")

    if {"tmID", "oppID", "W", "L"}.issubset(team_vs_team.columns):
        games = team_vs_team.groupby(["tmID", "oppID"]).agg({
            "W": "sum",
            "L": "sum"
        }).reset_index()

        games["matchup_balance"] = games["W"] - games["L"]
        save_table(games, "head_to_head.csv")


def recommend_similar_teams(df, team_id, top_n=5):
    features = [
        "win_rate",
        "goals_for_per_game",
        "goals_against_per_game",
        "goal_difference_per_game",
        "recommendation_score"
    ]

    profiles = df.groupby("tmID")[features].mean().dropna().reset_index()

    if team_id not in profiles["tmID"].values:
        raise ValueError(f"Команда {team_id} не найдена")

    values = profiles[features]
    normalized = (values - values.mean()) / values.std(ddof=0)

    target_index = profiles.index[profiles["tmID"] == team_id][0]
    target = normalized.loc[target_index]

    profiles["distance"] = np.sqrt(((normalized - target) ** 2).sum(axis=1))

    result = profiles[profiles["tmID"] != team_id].sort_values("distance").head(top_n)
    save_table(result, f"recommendations_for_{team_id}.csv")
    return result


def main():
    unpack_archive()

    teams = read_csv_fixed("Teams.csv")
    team_vs_team = read_csv_fixed("TeamVsTeam.csv")
    scoring = read_csv_fixed("Scoring.csv")
    goalies = read_csv_fixed("Goalies.csv")

    print("Teams.csv:", teams.shape)
    print("TeamVsTeam.csv:", team_vs_team.shape)
    print("Scoring.csv:", scoring.shape)
    print("Goalies.csv:", goalies.shape)

    teams = prepare_teams(teams)

    save_table(teams.head(10), "teams_head.csv")
    save_table(teams.describe(include="all").T.reset_index(), "teams_description.csv")

    missing_values(teams)
    teams = duplicates(teams)

    plot_histograms(teams)
    plot_seaborn(teams)
    plot_heatmaps(teams)
    plot_plotly(teams)

    outliers(teams)
    filters(teams)
    teams = add_noise(teams)

    categorical_analysis(teams)
    group_tables(teams, team_vs_team)

    if "tmID" in teams.columns:
        team_id = teams["tmID"].dropna().iloc[0]
        recommend_similar_teams(teams, team_id)

    save_table(teams, "teams_final.csv")

    print("Готово")
    print("Графики:", FIG_DIR)
    print("Таблицы:", TABLE_DIR)


if __name__ == "__main__":
    main()