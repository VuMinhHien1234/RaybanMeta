"""Dataset sources — mỗi source trả về cùng một cấu trúc `DataSource` với
3 split train/val/test, để stream + loader không phụ thuộc dataset cụ thể.

Có sẵn:
- resisc45  : NWPU-RESISC45, 31.500 ảnh trên không, 45 class (tải từ
              HuggingFace `timm/resisc45`, có sẵn split train/validation/test).
              Đây là dataset chính của G1 (gần ảnh UAV nhất trong các bộ dễ tải).
- eurosat   : 27.000 ảnh vệ tinh 64x64, 10 class (torchvision tự tải).
              Bộ nhẹ để chạy nhanh / debug pipeline.
- synthetic : ảnh giả sinh bằng numpy (mỗi class một màu trung bình riêng)
              — chạy smoke test không cần mạng.

Mọi class ở đây đều picklable (không dùng closure) để DataLoader num_workers>0
hoạt động trên cả macOS (spawn).
"""
# ↳ GIẢI THÍCH TỔNG QUAN: Mỗi bộ dữ liệu (RESISC45/EuroSAT/synthetic) có cách tải
#   khác nhau, nhưng file này "bọc" tất cả về CÙNG một hình dạng `DataSource`
#   (num_classes, class_names, 3 split). Nhờ vậy phần còn lại của code không cần
#   biết đang xài dataset nào -> dễ đổi dataset chỉ bằng sửa config.
# ↳ "picklable" = đối tượng copy được sang tiến trình con; cần thế để DataLoader
#   dùng nhiều worker (num_workers>0) không bị lỗi trên macOS.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np  # ↳ Dùng để tạo dữ liệu giả (synthetic) và xử lý mảng ảnh.

from .stream import stratified_split  # ↳ Mượn hàm chia tỉ lệ để tự tạo val/test cho EuroSAT.


# ----------------------------------------------------------------------------- splits
# ↳ 3 lớp "Split" dưới đây khác nhau ở CHỖ ẢNH NẰM Ở ĐÂU (đĩa / parquet / RAM),
#   nhưng đều có cùng 3 hàm: __len__, get_image(i), và thuộc tính labels.
class ImagePathSplit:
    """Split lưu (đường_dẫn, nhãn) — ảnh đọc từ đĩa khi cần."""
    # ↳ Nhẹ RAM: chỉ giữ đường dẫn, tới lúc cần mới mở ảnh từ đĩa. Dùng cho EuroSAT.

    def __init__(self, paths: List[str], labels: List[int]):
        assert len(paths) == len(labels)       # ↳ Mỗi ảnh phải có đúng 1 nhãn.
        self.paths = paths                     # ↳ Danh sách đường dẫn ảnh.
        self.labels = [int(y) for y in labels] # ↳ Ép nhãn về int cho chắc.

    def __len__(self) -> int:
        return len(self.labels)                # ↳ Số mẫu trong split.

    def get_image(self, i: int):
        from PIL import Image                   # ↳ Import trong hàm để file này không bắt buộc có PIL khi chỉ test logic.

        return Image.open(self.paths[int(i)]).convert("RGB")  # ↳ Mở ảnh thứ i, ép về 3 kênh màu RGB.


class HFSplit:
    """Split bọc quanh một split của HuggingFace `datasets` (ảnh trong parquet)."""
    # ↳ Dùng cho RESISC45: ảnh nằm trong dataset HuggingFace đã tải sẵn.

    def __init__(self, hf_dataset):
        self.ds = hf_dataset                             # ↳ Giữ tham chiếu tới dataset HF.
        self.labels = [int(y) for y in hf_dataset["label"]]  # ↳ Rút cột nhãn ra list để truy cập nhanh.

    def __len__(self) -> int:
        return len(self.labels)

    def get_image(self, i: int):
        return self.ds[int(i)]["image"].convert("RGB")  # ↳ Lấy ảnh thứ i từ HF, ép RGB.


class ArraySplit:
    """Split giữ ảnh trong RAM dưới dạng numpy uint8 (N, H, W, 3)."""
    # ↳ Dùng cho synthetic: mọi ảnh nằm luôn trong RAM (nhanh, không cần mạng/đĩa).

    def __init__(self, arrays: np.ndarray, labels: List[int]):
        assert len(arrays) == len(labels)
        self.arrays = arrays                    # ↳ Mảng numpy (N ảnh, cao, rộng, 3 kênh).
        self.labels = [int(y) for y in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def get_image(self, i: int):
        from PIL import Image

        return Image.fromarray(self.arrays[int(i)])  # ↳ Chuyển mảng numpy -> ảnh PIL để đi qua cùng đường transform.


@dataclass
class DataSource:
    # ↳ "Hộp chuẩn" mà mọi dataset trả về. Nhờ chuẩn hoá nên loader/stream dùng chung.
    name: str                       # ↳ Tên dataset.
    num_classes: int                # ↳ Tổng số class.
    class_names: List[str]          # ↳ Tên từng class (để log cho dễ đọc).
    splits: Dict[str, object]  # 'train' | 'val' | 'test' -> *Split
    # ↳ 3 khóa 'train'/'val'/'test', mỗi khóa là 1 đối tượng *Split ở trên.


# ----------------------------------------------------------------------------- sources
# ↳ Mỗi hàm _load_* dưới đây tải 1 dataset cụ thể rồi đóng gói thành DataSource.
def _load_resisc45(cfg: dict) -> DataSource:
    try:
        from datasets import load_dataset      # ↳ Thư viện HuggingFace `datasets`; có thể chưa cài.
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "RESISC45 cần thư viện HuggingFace `datasets`: pip install datasets"
        ) from e

    repo = cfg.get("hf_repo", "timm/resisc45")  # ↳ Kho dữ liệu trên HuggingFace (đổi được qua config).
    root = cfg.get("root", "./data")            # ↳ Thư mục lưu cache.
    ds = load_dataset(repo, cache_dir=str(Path(root) / "hf_cache"))  # ↳ Tải (hoặc lấy từ cache) dataset.
    feat = ds["train"].features["label"]        # ↳ Metadata của cột nhãn: biết số class và tên class.
    return DataSource(
        name="resisc45",
        num_classes=feat.num_classes,           # ↳ Số class lấy thẳng từ metadata.
        class_names=list(feat.names),           # ↳ Tên class lấy thẳng từ metadata.
        splits={
            "train": HFSplit(ds["train"]),       # ↳ RESISC45 có sẵn 3 split -> bọc thẳng.
            "val": HFSplit(ds["validation"]),
            "test": HFSplit(ds["test"]),
        },
    )


def _load_eurosat(cfg: dict) -> DataSource:
    from torchvision.datasets import EuroSAT  # cần torchvision  ↳ EuroSAT tải qua torchvision.

    root = cfg.get("root", "./data")
    base = EuroSAT(root=root, download=True)     # ↳ Tự tải EuroSAT nếu chưa có.
    paths = [p for p, _ in base.samples]         # ↳ base.samples là list (đường_dẫn, nhãn); tách riêng đường dẫn...
    labels = [int(y) for _, y in base.samples]   # ↳ ...và nhãn.
    seed = int(cfg.get("split_seed", 0))
    # Không có split chính thức -> tự tách 80/10/10 stratified, cố định theo seed.
    all_idx = list(range(len(labels)))           # ↳ Chỉ số 0..N-1 của toàn bộ ảnh.
    trainval, test = stratified_split(all_idx, labels, fraction=0.10, seed=seed)      # ↳ Cắt 10% ra làm test.
    train, val = stratified_split(trainval, labels, fraction=1 / 9, seed=seed + 1)    # ↳ Trong 90% còn lại, cắt 1/9 (~10% tổng) làm val.

    def take(idxs: List[int]) -> ImagePathSplit:
        # ↳ Hàm phụ: từ danh sách chỉ số -> tạo 1 split (đường dẫn + nhãn tương ứng).
        return ImagePathSplit([paths[i] for i in idxs], [labels[i] for i in idxs])

    return DataSource(
        name="eurosat",
        num_classes=len(base.classes),
        class_names=list(base.classes),
        splits={"train": take(train), "val": take(val), "test": take(test)},  # ↳ Gói 3 split tự chia.
    )


def _load_synthetic(cfg: dict) -> DataSource:
    """Ảnh giả học được: class c có màu trung bình riêng + nhiễu."""
    # ↳ Không cần mạng/đĩa: mỗi class = 1 màu nền riêng + nhiễu, model học được ngay
    #   -> dùng để kiểm tra pipeline chạy thông (smoke test), không phải để đo hay/dở.
    num_classes = int(cfg.get("num_classes", 6))       # ↳ Số class giả.
    n_train = int(cfg.get("train_per_class", 24))      # ↳ Số ảnh train mỗi class.
    n_val = int(cfg.get("val_per_class", 8))
    n_test = int(cfg.get("test_per_class", 8))
    size = int(cfg.get("image_size", 32))              # ↳ Cạnh ảnh vuông (px).
    rng = np.random.default_rng(int(cfg.get("split_seed", 0)))  # ↳ Bộ ngẫu nhiên cố định.
    means = rng.integers(40, 216, size=(num_classes, 3))        # ↳ Mỗi class 1 màu trung bình (R,G,B) khác nhau.

    def make(n_per_class: int, salt: int) -> ArraySplit:
        # ↳ Sinh 1 split: mỗi class n_per_class ảnh = màu class + nhiễu Gauss.
        r = np.random.default_rng(1000 + salt)         # ↳ "salt" khác nhau cho train/val/test để 3 tập không trùng.
        imgs, labels = [], []
        for c in range(num_classes):
            noise = r.normal(0, 25, size=(n_per_class, size, size, 3))  # ↳ Nhiễu quanh màu nền (lệch chuẩn 25).
            x = np.clip(means[c][None, None, None, :] + noise, 0, 255).astype(np.uint8)
            # ↳ Cộng màu nền + nhiễu, kẹp về [0,255], ép uint8 (đúng định dạng ảnh).
            imgs.append(x)
            labels += [c] * n_per_class               # ↳ Gắn nhãn c cho cả lô ảnh vừa tạo.
        return ArraySplit(np.concatenate(imgs, axis=0), labels)  # ↳ Nối mọi class thành 1 mảng lớn.

    return DataSource(
        name="synthetic",
        num_classes=num_classes,
        class_names=[f"class_{c}" for c in range(num_classes)],   # ↳ Đặt tên tạm "class_0", "class_1"...
        splits={"train": make(n_train, 1), "val": make(n_val, 2), "test": make(n_test, 3)},
    )


_REGISTRY = {
    # ↳ "Sổ đăng ký": ánh xạ tên dataset -> hàm tải tương ứng. Thêm dataset mới chỉ cần thêm 1 dòng ở đây.
    "resisc45": _load_resisc45,
    "eurosat": _load_eurosat,
    "synthetic": _load_synthetic,
}


def get_source(data_cfg: dict) -> DataSource:
    """Điểm vào duy nhất: get_source(cfg['data']) -> DataSource."""
    # ↳ Nơi duy nhất phần còn lại của code gọi để lấy dữ liệu. Đọc cfg['data']['name'] rồi chọn hàm tải.
    name = str(data_cfg.get("name", "")).lower()   # ↳ Lấy tên dataset, hạ chữ thường cho chắc.
    if name not in _REGISTRY:                      # ↳ Tên lạ -> báo lỗi kèm danh sách hợp lệ.
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](data_cfg)               # ↳ Gọi đúng hàm tải và trả về DataSource.
