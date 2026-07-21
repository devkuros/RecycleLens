"""Dataset loaders for labeled train images and unlabeled test images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def resolve_image_path(test_dir: Path, image_id: str) -> Path:
    """
    Resolve a template id to a file under test_dir.

    Supports bare filenames, numeric ids with common extensions,
    and zero-padded variants (e.g. 1 → 0001.jpg).
    """
    test_dir = Path(test_dir)
    direct = test_dir / image_id
    if direct.is_file():
        return direct

    candidates: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        candidates.append(test_dir / f"{image_id}{ext}")
        if image_id.isdigit():
            candidates.append(test_dir / f"{int(image_id):04d}{ext}")
            candidates.append(test_dir / f"{int(image_id):05d}{ext}")

    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"Test image for id '{image_id}' not found under {test_dir}"
    )


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _class_id_from_folder_name(name: str) -> int | None:
    """
    Map a train subfolder name to class id.

    Accepts: 0, 1, 2, Recyclable, Electronic, Organic,
    and prefixed forms like 0_Recyclable / 1_Electronic / 2_Organic.
    """
    key = name.lower().strip()
    if key in {"0", "1", "2"}:
        return int(key)
    if len(key) >= 2 and key[0] in "012" and key[1] in "_- .":
        return int(key[0])

    aliases = (
        (("recyclable", "recycle"), 0),
        (("electronic", "e-waste", "ewaste"), 1),
        (("organic",), 2),
    )
    for names, class_id in aliases:
        if any(token in key for token in names):
            return class_id
    return None


def discover_train_samples(train_dir: Path) -> tuple[list[Path], list[int]]:
    """
    Discover train images from class subfolders.

    Supports train/{0,1,2}/, named folders, and prefixed names
    like 0_Recyclable / 1_Electronic / 2_Organic.

    Returns parallel lists of paths and integer labels.
    """
    train_dir = Path(train_dir)
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    paths: list[Path] = []
    labels: list[int] = []

    for folder in sorted(train_dir.iterdir()):
        if not folder.is_dir():
            continue
        class_id = _class_id_from_folder_name(folder.name)
        if class_id is None:
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(path)
                labels.append(class_id)

    if not paths:
        raise FileNotFoundError(
            f"No labeled training images found under {train_dir}. "
            "Expected subfolders 0/, 1/, 2/ "
            "(or Recyclable/Electronic/Organic, or 0_Recyclable/…)."
        )
    return paths, labels


class WasteTrainDataset(Dataset):
    """Labeled waste images for training / validation folds."""

    def __init__(
        self,
        paths: list[Path],
        labels: list[int],
        transform=None,
    ) -> None:
        if len(paths) != len(labels):
            raise ValueError("paths and labels must have the same length")
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        label = self.labels[index]
        image = _read_rgb(path)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return {"image": image, "label": label, "path": str(path)}


class WasteTestDataset(Dataset):
    """
    Unlabeled test images ordered by template.csv.

    Guardrail: this dataset is for inference only — no labels are loaded.
    """

    def __init__(
        self,
        template_csv: Path,
        test_dir: Path,
        id_column: str = "id",
        transform=None,
    ) -> None:
        self.test_dir = Path(test_dir)
        self.transform = transform
        df = pd.read_csv(template_csv)
        if id_column not in df.columns:
            raise ValueError(f"Column '{id_column}' not found in {template_csv}")
        self.ids = df[id_column].astype(str).tolist()
        self.paths = [resolve_image_path(self.test_dir, image_id) for image_id in self.ids]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        image_id = self.ids[index]
        image = _read_rgb(path)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return {"image": image, "id": image_id, "path": str(path)}
