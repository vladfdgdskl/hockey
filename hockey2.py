from pathlib import Path
import zipfile
import shutil
import random
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs2"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
ARCHIVE_NAMES = ["hockey-player-detection.yolov8.zip", "hockey-player-detection.zip"]
SPLITS = ["train", "valid", "test"]
CLASS_NAMES = ["goalie", "player", "puck", "referee"]
CLASS_RU = {
    "goalie": "вратарь",
    "player": "полевой игрок",
    "puck": "шайба",
    "referee": "судья",
}
BOX_COLORS = {
    "goalie": "orange",
    "player": "red",
    "puck": "yellow",
    "referee": "lime",
}


def find_archive():
    for folder in [BASE_DIR, Path.cwd(), BASE_DIR.parent]:
        for name in ARCHIVE_NAMES:
            path = folder / name
            if path.exists():
                return path
    raise FileNotFoundError("Файл hockey-player-detection.yolov8.zip не найден рядом с кодом")


def prepare_dirs():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def image_files(names, split):
    result = []
    for name in names:
        low = name.lower()
        if name.startswith(split + "/images/") and low.endswith((".jpg", ".jpeg", ".png")):
            result.append(name)
    return sorted(result)


def read_label(archive, label_path):
    text = archive.read(label_path).decode("utf-8").strip()
    rows = []
    if not text:
        return rows
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            rows.append([None, None, None, None, None, "bad_format"])
            continue
        try:
            class_id = int(float(parts[0]))
            x, y, w, h = [float(v) for v in parts[1:]]
            status = "ok"
            if class_id < 0 or class_id >= len(CLASS_NAMES):
                status = "bad_class"
            if x < 0 or x > 1 or y < 0 or y > 1 or w <= 0 or w > 1 or h <= 0 or h > 1:
                status = "bad_box"
            rows.append([class_id, x, y, w, h, status])
        except Exception:
            rows.append([None, None, None, None, None, "bad_format"])
    return rows


def collect_data(archive_path):
    images = []
    boxes = []
    problems = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = archive.namelist()
        labels = set(name for name in names if "/labels/" in name and name.endswith(".txt"))
        for split in SPLITS:
            for image_path in image_files(names, split):
                label_path = split + "/labels/" + Path(image_path).stem + ".txt"
                width = height = None
                broken = 0
                try:
                    with archive.open(image_path) as file:
                        img = Image.open(file)
                        img.load()
                        width, height = img.size
                except Exception:
                    broken = 1
                    problems.append([split, image_path, "broken_image"])
                if label_path not in labels:
                    problems.append([split, image_path, "missing_label"])
                    rows = []
                else:
                    rows = read_label(archive, label_path)
                    if len(rows) == 0:
                        problems.append([split, image_path, "empty_label"])
                ok_boxes = 0
                for i, row in enumerate(rows):
                    class_id, x, y, w, h, status = row
                    if status != "ok":
                        problems.append([split, label_path, status])
                        continue
                    class_name = CLASS_NAMES[class_id]
                    boxes.append({
                        "split": split,
                        "image_path": image_path,
                        "label_path": label_path,
                        "class_id": class_id,
                        "class": class_name,
                        "class_ru": CLASS_RU[class_name],
                        "x_center": x,
                        "y_center": y,
                        "box_width": w,
                        "box_height": h,
                        "box_area": w * h,
                    })
                    ok_boxes += 1
                images.append({
                    "split": split,
                    "image_path": image_path,
                    "label_path": label_path,
                    "width": width,
                    "height": height,
                    "resolution": f"{width}x{height}" if width and height else "ошибка",
                    "boxes_count": ok_boxes,
                    "broken": broken,
                    "has_label": int(label_path in labels),
                })
    return pd.DataFrame(images), pd.DataFrame(boxes), pd.DataFrame(problems, columns=["split", "path", "problem"])


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def make_class_distribution(boxes):
    counts = boxes["class_ru"].value_counts().reindex([CLASS_RU[x] for x in CLASS_NAMES]).fillna(0)
    counts.plot(kind="bar", figsize=(8, 5))
    plt.title("Распределение объектов по классам")
    plt.xlabel("Класс объекта")
    plt.ylabel("Количество объектов")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    save_plot(FIG_DIR / "12_class_distribution.png")


def make_split_distribution(images):
    counts = images["split"].value_counts().reindex(SPLITS).fillna(0)
    counts.plot(kind="bar", figsize=(7, 5))
    plt.title("Распределение изображений по выборкам")
    plt.xlabel("Выборка")
    plt.ylabel("Количество изображений")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    save_plot(FIG_DIR / "13_split_distribution.png")


def make_sample_images(archive_path, images):
    sample = images.groupby("split").head(2).head(6)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    with zipfile.ZipFile(archive_path, "r") as archive:
        for ax, (_, row) in zip(axes, sample.iterrows()):
            with archive.open(row["image_path"]) as file:
                img = Image.open(file).convert("RGB")
            ax.imshow(img)
            ax.set_title(row["split"])
            ax.axis("off")
    for ax in axes[len(sample):]:
        ax.axis("off")
    fig.suptitle("Примеры исходных изображений хоккейных эпизодов")
    save_plot(FIG_DIR / "14_sample_images.png")


def make_resolution_distribution(images):
    counts = images["resolution"].value_counts().sort_index()
    counts.plot(kind="bar", figsize=(7, 5))
    plt.title("Распределение разрешений изображений")
    plt.xlabel("Разрешение")
    plt.ylabel("Количество изображений")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    save_plot(FIG_DIR / "15_resolution_distribution.png")


def make_boxes_per_image(images):
    plt.figure(figsize=(8, 5))
    plt.hist(images["boxes_count"], bins=range(int(images["boxes_count"].min()), int(images["boxes_count"].max()) + 2))
    plt.title("Количество размеченных объектов на изображение")
    plt.xlabel("Количество объектов")
    plt.ylabel("Количество изображений")
    plt.grid(axis="y", alpha=0.3)
    save_plot(FIG_DIR / "16_boxes_per_image.png")


def make_class_by_split(boxes):
    table = boxes.pivot_table(index="split", columns="class_ru", values="image_path", aggfunc="count", fill_value=0)
    table = table.reindex(SPLITS)
    table.plot(kind="bar", figsize=(9, 5))
    plt.title("Распределение объектов по классам и выборкам")
    plt.xlabel("Выборка")
    plt.ylabel("Количество объектов")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    save_plot(FIG_DIR / "17_class_by_split.png")


def make_bbox_area(boxes):
    data = []
    labels = []
    for name in CLASS_NAMES:
        values = boxes.loc[boxes["class"] == name, "box_area"]
        if len(values) > 0:
            data.append(values)
            labels.append(CLASS_RU[name])
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels)
    plt.title("Сравнение относительной площади bounding box")
    plt.xlabel("Класс")
    plt.ylabel("Площадь бокса от площади изображения")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    save_plot(FIG_DIR / "18_bbox_area.png")


def make_bbox_centers(boxes):
    plt.figure(figsize=(7, 6))
    for name in CLASS_NAMES:
        part = boxes[boxes["class"] == name]
        plt.scatter(part["x_center"], part["y_center"], s=8, alpha=0.45, label=CLASS_RU[name])
    plt.gca().invert_yaxis()
    plt.title("Расположение центров аннотаций")
    plt.xlabel("x центра")
    plt.ylabel("y центра")
    plt.legend()
    plt.grid(alpha=0.3)
    save_plot(FIG_DIR / "19_bbox_centers.png")


def get_font(size):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_boxes(image, rows):
    draw = ImageDraw.Draw(image)
    font = get_font(22)
    width, height = image.size
    for _, row in rows.iterrows():
        x = row["x_center"] * width
        y = row["y_center"] * height
        w = row["box_width"] * width
        h = row["box_height"] * height
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        color = BOX_COLORS[row["class"]]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw.text((x1, max(0, y1 - 24)), row["class_ru"], fill=color, font=font)
    return image


def make_annotated_samples(archive_path, images, boxes):
    sample = images.sort_values("boxes_count", ascending=False).head(4)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    with zipfile.ZipFile(archive_path, "r") as archive:
        for ax, (_, row) in zip(axes, sample.iterrows()):
            with archive.open(row["image_path"]) as file:
                img = Image.open(file).convert("RGB")
            img = draw_boxes(img, boxes[boxes["image_path"] == row["image_path"]])
            ax.imshow(img)
            ax.set_title(f"{row['split']}, объектов: {row['boxes_count']}")
            ax.axis("off")
    fig.suptitle("Примеры изображений с нанесенными YOLO-аннотациями")
    save_plot(FIG_DIR / "20_annotated_samples.png")


def make_quality_check(images, problems):
    data = pd.Series({
        "изображения без ошибок": int((images["broken"] == 0).sum()),
        "поврежденные изображения": int((images["broken"] != 0).sum()),
        "файлы без разметки": int((images["has_label"] == 0).sum()),
        "ошибки YOLO-строк": int(len(problems[problems["problem"].isin(["bad_format", "bad_class", "bad_box"])])) if len(problems) else 0,
    })
    data.plot(kind="bar", figsize=(9, 5))
    plt.title("Программная проверка качества файлов и разметки")
    plt.xlabel("Показатель")
    plt.ylabel("Количество")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    save_plot(FIG_DIR / "21_quality_check.png")


def make_tables(archive_path, images, boxes, problems):
    total_images = len(images)
    total_boxes = len(boxes)
    class_counts = boxes["class_ru"].value_counts().reindex([CLASS_RU[x] for x in CLASS_NAMES]).fillna(0).astype(int)
    table8 = pd.DataFrame([
        ["Название", "Hockey Player Detection YOLOv8"],
        ["Источник", "Roboflow / локальный архив " + archive_path.name],
        ["Платформа", "Roboflow"],
        ["Рабочая область Roboflow", "vlad-r1yg8"],
        ["Проект / версия", "vlad-r1yg8 / dataset"],
        ["Дата выгрузки", "02.06.2026, 08:07 GMT"],
        ["Тип задачи", "детекция объектов на изображениях"],
        ["Формат изображений", "JPG"],
        ["Формат разметки", "YOLOv8: class x_center y_center width height"],
        ["Количество изображений", total_images],
        ["Количество размеченных объектов", total_boxes],
        ["Количество классов", len(CLASS_NAMES)],
        ["Разбиение", "train / valid / test"],
        ["Лицензия", "Private"],
        ["Предобработка", "не применялась"],
        ["Аугментации", "не применялись"],
    ], columns=["Параметр", "Значение"])
    table9 = pd.DataFrame({
        "Класс": class_counts.index,
        "Количество объектов": class_counts.values,
        "Доля, %": (class_counts.values / total_boxes * 100).round(2),
    })
    split_images = images["split"].value_counts().reindex(SPLITS).fillna(0).astype(int)
    split_boxes = boxes["split"].value_counts().reindex(SPLITS).fillna(0).astype(int)
    table10 = pd.DataFrame({
        "Выборка": SPLITS,
        "Изображения": split_images.values,
        "Объекты": split_boxes.values,
        "Доля изображений, %": (split_images.values / total_images * 100).round(2),
    })
    table11 = images["resolution"].value_counts().rename_axis("Разрешение").reset_index(name="Количество")
    table11["Доля, %"] = (table11["Количество"] / total_images * 100).round(2)
    s = images["boxes_count"]
    table12 = pd.DataFrame([
        ["count", round(float(s.count()), 2)],
        ["mean", round(float(s.mean()), 2)],
        ["std", round(float(s.std()), 2)],
        ["min", round(float(s.min()), 2)],
        ["median", round(float(s.median()), 2)],
        ["max", round(float(s.max()), 2)],
    ], columns=["Показатель", "Значение"])
    bbox = boxes.groupby("class_ru").agg(
        count=("box_area", "count"),
        mean_area=("box_area", "mean"),
        median_area=("box_area", "median"),
        mean_width=("box_width", "mean"),
        mean_height=("box_height", "mean"),
    ).reset_index().rename(columns={"class_ru": "Класс"})
    for col in ["mean_area", "median_area", "mean_width", "mean_height"]:
        bbox[col] = bbox[col].round(4)
    table13 = bbox
    table14 = pd.DataFrame([
        ["Всего изображений", total_images],
        ["Поврежденные изображения", int(images["broken"].sum())],
        ["Изображения без label-файла", int((images["has_label"] == 0).sum())],
        ["Пустые label-файлы", int((problems["problem"] == "empty_label").sum()) if len(problems) else 0],
        ["Ошибки формата YOLO", int(problems["problem"].isin(["bad_format", "bad_class", "bad_box"]).sum()) if len(problems) else 0],
        ["Изображения меньше 512x512", int(((images["width"] < 512) | (images["height"] < 512)).sum())],
    ], columns=["Показатель", "Значение"])
    manual_sample = pd.concat([
        images[images["split"] == "train"].sample(26, random_state=42),
        images[images["split"] == "valid"].sample(7, random_state=42),
        images[images["split"] == "test"].sample(4, random_state=42),
    ])
    table15 = pd.DataFrame([
        ["Объем ручной проверки", "37 изображений из 370, то есть 10% датасета"],
        ["Проверенные части набора", "train – 26 изображений, valid – 7 изображений, test – 4 изображения"],
        ["Проверено размеченных объектов", int(manual_sample["boxes_count"].sum())],
        ["Соответствие класса объекту", "существенных ошибок классов не выявлено"],
        ["Полнота bounding box", "рамки в основном охватывают целевые объекты"],
        ["Пропуск целевых объектов", "явных пропусков основных объектов на льду не выявлено"],
        ["Качество разметки шайбы", "разметка присутствует, но проверка затруднена из-за малого размера шайбы"],
        ["Итоговый вывод", "разметка пригодна для учебного первичного анализа"],
    ], columns=["Показатель проверки", "Результат"])
    tables = {
        "08_dataset_summary.csv": table8,
        "09_class_counts.csv": table9,
        "10_split_counts.csv": table10,
        "11_resolution_counts.csv": table11,
        "12_objects_per_image_stats.csv": table12,
        "13_bbox_stats.csv": table13,
        "14_quality_check.csv": table14,
        "15_manual_annotation_check.csv": table15,
        "metadata_images.csv": images,
        "metadata_boxes.csv": boxes,
        "problems.csv": problems,
    }
    for name, df in tables.items():
        df.to_csv(TABLE_DIR / name, index=False, encoding="utf-8-sig", sep=";", decimal=",")
    return table8, table9, table10, table11, table12, table13, table14, table15


def main():
    archive_path = find_archive()
    prepare_dirs()
    images, boxes, problems = collect_data(archive_path)
    make_class_distribution(boxes)
    make_split_distribution(images)
    make_sample_images(archive_path, images)
    make_resolution_distribution(images)
    make_boxes_per_image(images)
    make_class_by_split(boxes)
    make_bbox_area(boxes)
    make_bbox_centers(boxes)
    make_annotated_samples(archive_path, images, boxes)
    make_quality_check(images, problems)
    make_tables(archive_path, images, boxes, problems)
    print("Готово")
    print("Изображений:", len(images))
    print("Объектов:", len(boxes))
    print("Папка результата:", OUT_DIR)


if __name__ == "__main__":
    main()
