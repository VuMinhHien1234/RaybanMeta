"""Torch DataLoaders cho từng task trong stream (phần này cần torch/torchvision)."""
# ↳ GIẢI THÍCH TỔNG QUAN: `DataLoader` của PyTorch là "băng chuyền" nạp ảnh theo
#   lô (batch) khi train. File này biến mỗi task (TaskSpec) thành 3 băng chuyền
#   train/val/test, kèm bước tiền xử lý ảnh (resize, chuẩn hoá, tăng cường dữ liệu).
from __future__ import annotations
from typing import Dict, List
import torch
from torch.utils.data import DataLoader, Dataset  # ↳ Dataset = nguồn 1 mẫu; DataLoader = gom mẫu thành batch.
from torchvision import transforms as T           # ↳ Bộ phép biến đổi ảnh dựng sẵn.

from .sources import DataSource
from .stream import TaskSpec

IMAGENET_MEAN = (0.485, 0.456, 0.406)  # ↳ Trung bình màu của ImageNet — backbone pretrained mong dữ liệu chuẩn hoá theo bộ này.
IMAGENET_STD = (0.229, 0.224, 0.225)   # ↳ Độ lệch chuẩn màu ImageNet (dùng cùng mục đích trên).


def build_transforms(image_size: int, train: bool):
    """Aug nhẹ, chuẩn ImageNet (khớp backbone pretrained của timm)."""
    # ↳ "Aug" (augmentation) = biến đổi ngẫu nhiên ảnh train để model bớt học vẹt.
    #   Khi eval (val/test) thì KHÔNG aug ngẫu nhiên, chỉ resize + chuẩn hoá.
    if train:
        return T.Compose(                         # ↳ Compose = xâu chuỗi nhiều phép biến đổi.
            [
                T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),  # ↳ Cắt ngẫu nhiên 60-100% ảnh rồi resize -> đa dạng góc nhìn.
                T.RandomHorizontalFlip(),          # ↳ Lật ngang ngẫu nhiên.
                T.ToTensor(),                      # ↳ Ảnh PIL -> tensor, giá trị về [0,1].
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # ↳ Chuẩn hoá theo thống kê ImageNet.
            ]
        )
    return T.Compose(
        [
            T.Resize((image_size, image_size)),    # ↳ Eval: chỉ resize cố định, không cắt ngẫu nhiên.
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class TaskDataset(Dataset):
    """Một split của một task: đọc ảnh qua source-split, trả (tensor, nhãn_gốc).

    Nhãn giữ nguyên id toàn cục (không remap) — head có đủ cột cho mọi class,
    việc giới hạn class nào được dùng do mask logits ở engine quyết định.
    """
    # ↳ Điểm mấu chốt: nhãn KHÔNG bị đánh số lại theo task. Head luôn có đủ cột cho
    #   toàn bộ class; muốn "chỉ xét class của task hiện tại" thì engine sẽ che
    #   (mask) các cột khác — chứ không đổi nhãn ở đây.

    def __init__(self, split, indices: List[int], transform):
        self.split = split          # ↳ Nguồn ảnh (một *Split từ sources.py).
        self.indices = list(indices)  # ↳ Chỉ những mẫu thuộc task này.
        self.transform = transform  # ↳ Phép tiền xử lý áp cho mỗi ảnh.

    def __len__(self) -> int:
        return len(self.indices)    # ↳ DataLoader hỏi "có bao nhiêu mẫu?".

    def __getitem__(self, k: int):
        # ↳ DataLoader hỏi "mẫu thứ k là gì?". k chạy 0..len-1 trong task này.
        i = self.indices[k]                       # ↳ Đổi chỉ số cục bộ (k) sang chỉ số thật trong split (i).
        x = self.transform(self.split.get_image(i))  # ↳ Đọc ảnh i rồi chạy qua transform -> tensor.
        y = int(self.split.labels[i])             # ↳ Lấy nhãn gốc (toàn cục) của ảnh.
        return x, y                               # ↳ Trả (ảnh_tensor, nhãn).


def build_task_loaders(
    source: DataSource, stream: List[TaskSpec], data_cfg: dict
) -> List[Dict[str, DataLoader]]:
    """Trả về, cho mỗi task, dict {'train','val','test'} DataLoader."""
    image_size = int(data_cfg.get("image_size", 224))  # ↳ Kích cỡ ảnh đưa vào model (224 chuẩn ViT/timm).
    batch_size = int(data_cfg.get("batch_size", 32))   # ↳ Số ảnh mỗi lô.
    num_workers = int(data_cfg.get("num_workers", 2))  # ↳ Số tiến trình nạp ảnh song song.
    tf_train = build_transforms(image_size, train=True)   # ↳ Transform cho train (có aug).
    tf_eval = build_transforms(image_size, train=False)   # ↳ Transform cho val/test (không aug).

    # --- D6: TRÔI điều kiện quan sát (mặc định TẮT -> đi đúng đường cũ, bất biến) --------
    # Khi bật, mỗi task có transform RIÊNG với mức trôi tăng dần. Bản gốc dựng transform
    # MỘT LẦN cho mọi task (2 dòng trên), nên phải rẽ nhánh ở đây.
    drift_cfg = dict(data_cfg.get("drift") or {})
    drift_bat = bool(drift_cfg.get("enabled", False))
    ap_cho = set(drift_cfg.get("apply_to", ["train", "val", "test"]))
    seed_drift = int(data_cfg.get("seed", 0))
    if drift_bat:
        from .drift import bang_muc_troi, build_drift_transform
        print(f"[drift] BẬT · mode={drift_cfg.get('mode', 'linear')} "
              f"severity={drift_cfg.get('severity', 1.0)} "
              f"jitter={drift_cfg.get('jitter', 0.15)} · áp cho {sorted(ap_cho)}")
        print(f"[drift] mức từng task: {bang_muc_troi(len(stream), drift_cfg)}")

    def tf_cua_task(t: int, split_name: str, train: bool):
        """Transform của task t. Không bật drift -> trả đúng transform cũ."""
        if not drift_bat or split_name not in ap_cho:
            return tf_train if train else tf_eval
        return build_drift_transform(image_size, t, len(stream), drift_cfg,
                                     train=train, seed=seed_drift)

    def dl(split_name: str, idx: List[int], train: bool, t: int) -> DataLoader:
        # ↳ Hàm phụ dựng 1 DataLoader từ tên split + danh sách chỉ số.
        ds = TaskDataset(source.splits[split_name], idx, tf_cua_task(t, split_name, train))
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=train,                         # ↳ Chỉ xáo khi train; val/test giữ thứ tự cố định.
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),  # ↳ Có GPU thì "ghim" RAM để copy lên GPU nhanh hơn.
            drop_last=False,                       # ↳ Giữ lại lô cuối dù thiếu (không bỏ mẫu nào).
        )

    out = []
    for t, spec in enumerate(stream):              # ↳ Với mỗi task trong stream...
        out.append(
            {
                "train": dl("train", spec.train_idx, True, t),   # ↳ ...tạo 3 loader tương ứng.
                "val": dl("val", spec.val_idx, False, t),
                "test": dl("test", spec.test_idx, False, t),
            }
        )
    return out                                     # ↳ Danh sách: phần tử t = bộ loader của task t.


def build_eval_loader(source: DataSource, indices: List[int], data_cfg: dict,
                      split_name: str = "test") -> DataLoader:
    """#27 — DataLoader eval (không aug, không xáo) trên danh sách chỉ số tuỳ ý của 1 split.

    Dùng cho open-set: gom mẫu test của các class GIỮ LẠI (chưa từng train) thành 1 loader."""
    image_size = int(data_cfg.get("image_size", 224))
    batch_size = int(data_cfg.get("eval_batch_size", data_cfg.get("batch_size", 32)))
    num_workers = int(data_cfg.get("num_workers", 2))
    ds = TaskDataset(source.splits[split_name], indices, build_transforms(image_size, train=False))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False)
