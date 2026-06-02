from pathlib import Path
import io
import shutil
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs4_manual"
DATA_DIR = OUT_DIR / "data"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

DATASET_NAMES = [
    "dataset4glava",
    "dataset4glava.zip",

]

LABELS = ["weak_season", "medium_season", "strong_season"]
LABEL_RU = {
    "weak_season": "слабый сезон",
    "medium_season": "средний сезон",
    "strong_season": "сильный сезон",
}


def find_dataset():
    folders = [BASE_DIR, Path.cwd(), BASE_DIR.parent]
    for folder in folders:
        for name in DATASET_NAMES:
            path = folder / name
            if path.exists():
                return path
    raise FileNotFoundError("Не найден датасет. Положите рядом с кодом папку dataset4glava или архив dataset4glava.zip")


def create_folders():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    DATA_DIR.mkdir(parents=True)
    TABLE_DIR.mkdir(parents=True)
    FIG_DIR.mkdir(parents=True)


def read_csv_text(text):
    return pd.read_csv(io.StringIO(text), encoding="utf-8-sig", sep=None, engine="python")


def read_dataset(path):
    if path.is_dir():
        files = list(path.rglob("full_dataset.csv"))
        if not files:
            raise FileNotFoundError("В папке датасета не найден full_dataset.csv")
        return pd.read_csv(files[0], encoding="utf-8-sig", sep=None, engine="python")
    with zipfile.ZipFile(path, "r") as archive:
        name = next(x for x in archive.namelist() if x.endswith("full_dataset.csv"))
        text = archive.read(name).decode("utf-8-sig")
        return read_csv_text(text)


def copy_file_to_output(name, data):
    target = DATA_DIR / Path(name).name if "/data/" in name.replace("\\", "/") or name.endswith(".xlsx") else OUT_DIR / Path(name).name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def copy_main_files(path):
    needed = [
        "README.md",
        "DATASET_CARD.md",
        "full_dataset.csv",
        "classification_minimal.csv",
        "train.csv",
        "val.csv",
        "test.csv",
        "label_mapping.csv",
        "examples_sample.csv",
        "hockey_text_dataset_ch4_rovno.xlsx",
        "hockey_text_dataset_ch4.xlsx",
    ]
    if path.is_dir():
        for file in path.rglob("*"):
            if file.is_file() and file.name in needed:
                data = file.read_bytes()
                short_name = "data/" + file.name if file.suffix in [".csv", ".xlsx"] else file.name
                copy_file_to_output(short_name, data)
    else:
        with zipfile.ZipFile(path, "r") as archive:
            for name in archive.namelist():
                if Path(name).name in needed:
                    copy_file_to_output(name, archive.read(name))


def label_by_win_rate(value):
    if value <= 0.35:
        return "weak_season"
    if value <= 0.55:
        return "medium_season"
    return "strong_season"


def save_table(df, name):
    df.to_csv(TABLE_DIR / name, index=False, sep=";", encoding="utf-8-sig")


def save_plot(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=300, bbox_inches="tight")
    plt.close()


def prepare_columns(df):
    if "text_word_count" not in df.columns:
        df["text_word_count"] = df["text"].astype(str).str.split().str.len()
    if "char_count" not in df.columns:
        df["char_count"] = df["text"].astype(str).str.len()
    if "label_ru" not in df.columns:
        df["label_ru"] = df["label"].map(LABEL_RU)
    if "text_style" not in df.columns:
        df["text_style"] = "не указан"
    return df


def build_tables(df):
    df = prepare_columns(df)
    class_table = df.groupby(["label", "label_ru"]).size().reset_index(name="count")
    class_table["percent"] = (class_table["count"] / len(df) * 100).round(2)
    class_table["sort"] = class_table["label"].map({label: i for i, label in enumerate(LABELS)})
    class_table = class_table.sort_values("sort").drop(columns="sort")

    split_table = df.pivot_table(index="split", columns="label_ru", values="id", aggfunc="count", fill_value=0)
    split_table = split_table.reindex(["train", "val", "test"])
    for label in ["слабый сезон", "средний сезон", "сильный сезон"]:
        if label not in split_table.columns:
            split_table[label] = 0
    split_table = split_table[["слабый сезон", "средний сезон", "сильный сезон"]]
    split_table["Итого"] = split_table.sum(axis=1)
    split_table = split_table.reset_index().rename(columns={"split": "Выборка"})

    length_table = pd.DataFrame([
        ["Количество слов", df["text_word_count"].count(), df["text_word_count"].mean(), df["text_word_count"].min(), df["text_word_count"].median(), df["text_word_count"].max()],
        ["Количество символов", df["char_count"].count(), df["char_count"].mean(), df["char_count"].min(), df["char_count"].median(), df["char_count"].max()],
    ], columns=["Показатель", "count", "mean", "min", "median", "max"]).round(2)

    style_table = df["text_style"].value_counts().rename_axis("text_style").reset_index(name="count")

    label_errors = 0
    if "win_rate" in df.columns:
        label_errors = int((df["label"] != df["win_rate"].apply(label_by_win_rate)).sum())

    quality_table = pd.DataFrame([
        ["Всего записей", len(df)],
        ["Пропуски в text", int(df["text"].isna().sum())],
        ["Пропуски в label", int(df["label"].isna().sum())],
        ["Дубликаты text", int(df.duplicated("text").sum())],
        ["Ошибки меток по win_rate", label_errors],
        ["Найдено email", int(df["text"].astype(str).str.contains(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", regex=True).sum())],
        ["Найдено телефонов", int(df["text"].astype(str).str.contains(r"(?:\+?\d[\d\-\s()]{8,}\d)", regex=True).sum())],
    ], columns=["Показатель", "Значение"])

    examples = []
    for label in LABELS:
        part = df[df["label"] == label]
        if len(part):
            row = part.iloc[0]
            text = str(row["text"])
            examples.append([LABEL_RU[label], row["text_style"], text[:260] + ("..." if len(text) > 260 else "")])
    examples_table = pd.DataFrame(examples, columns=["Класс", "Стиль", "Пример текста"])

    return class_table, split_table, length_table, style_table, quality_table, examples_table


def build_plots(df, class_table, split_table, style_table, quality_table):
    plt.figure(figsize=(8, 5))
    plt.bar(class_table["label_ru"], class_table["count"])
    plt.title("Распределение текстов по классам")
    plt.xlabel("Класс")
    plt.ylabel("Количество текстов")
    plt.grid(axis="y", alpha=0.3)
    save_plot("27_class_distribution.png")

    split_table.set_index("Выборка")[["слабый сезон", "средний сезон", "сильный сезон"]].plot(kind="bar", figsize=(8, 5))
    plt.title("Распределение текстов по выборкам")
    plt.xlabel("Выборка")
    plt.ylabel("Количество текстов")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    save_plot("28_split_distribution.png")

    plt.figure(figsize=(8, 5))
    plt.hist(df["text_word_count"], bins=25)
    plt.title("Распределение длины текстов")
    plt.xlabel("Количество слов")
    plt.ylabel("Количество текстов")
    plt.grid(axis="y", alpha=0.3)
    save_plot("29_text_length_distribution.png")

    top_styles = style_table.head(12).sort_values("count")
    plt.figure(figsize=(10, 6))
    plt.barh(top_styles["text_style"], top_styles["count"])
    plt.title("Распределение текстовых шаблонов")
    plt.xlabel("Количество текстов")
    plt.ylabel("Стиль текста")
    plt.grid(axis="x", alpha=0.3)
    save_plot("30_style_distribution.png")

    total = int(quality_table.loc[quality_table["Показатель"] == "Всего записей", "Значение"].iloc[0])
    issues = int(quality_table.loc[quality_table["Показатель"] != "Всего записей", "Значение"].sum())
    plt.figure(figsize=(7, 5))
    plt.bar(["корректные записи", "найденные проблемы"], [total - min(total, issues), issues])
    plt.title("Проверка качества текстового датасета")
    plt.xlabel("Результат проверки")
    plt.ylabel("Количество")
    plt.grid(axis="y", alpha=0.3)
    save_plot("31_quality_check.png")


def main():
    dataset_path = find_dataset()
    create_folders()
    copy_main_files(dataset_path)
    df = read_dataset(dataset_path)
    df = prepare_columns(df)

    tables = build_tables(df)
    names = [
        "class_distribution.csv",
        "split_distribution.csv",
        "text_length_stats.csv",
        "style_distribution.csv",
        "quality_check.csv",
        "examples_for_report.csv",
    ]
    for table, name in zip(tables, names):
        save_table(table, name)

    build_plots(df, tables[0], tables[1], tables[3], tables[4])

    print("Готово")
    print("Датасет:", dataset_path)
    print("Записей:", len(df))
    print("Результаты:", OUT_DIR)


if __name__ == "__main__":
    main()
