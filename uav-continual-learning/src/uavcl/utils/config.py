"""Đọc/ghi config yaml — mọi thí nghiệm chạy từ 1 file yaml + (tùy chọn) override."""
# ↳ GIẢI THÍCH TỔNG QUAN FILE NÀY:
#   Mọi thí nghiệm (G1..G4) được mô tả bằng 1 file .yaml (dataset nào, model nào,
#   learning rate bao nhiêu...). File này chỉ làm 3 việc: (1) đọc yaml -> dict,
#   (2) ghi dict -> yaml, (3) cho phép sửa nhanh 1 giá trị từ dòng lệnh bằng
#   cú pháp `--set a.b=c` mà không cần mở file yaml ra sửa tay.
from __future__ import annotations  # ↳ Cho phép dùng gợi ý kiểu mới (vd `str | Path`) trên Python cũ.

from pathlib import Path  # ↳ Path = cách làm việc với đường dẫn file an toàn, đa hệ điều hành.

import yaml  # ↳ Thư viện đọc/ghi định dạng YAML (config người-đọc-được).


def load_config(path: str | Path) -> dict:
    # ↳ Nhận đường dẫn file yaml, trả về 1 dict Python.
    with open(path, "r", encoding="utf-8") as f:  # ↳ Mở file ở chế độ đọc, mã UTF-8 (đọc được tiếng Việt).
        cfg = yaml.safe_load(f)                    # ↳ safe_load: parse yaml an toàn (không chạy code lạ trong file).
    if not isinstance(cfg, dict):                  # ↳ Config bắt buộc phải là 1 "mapping" (khối key: value).
        raise ValueError(f"Config {path} must be a yaml mapping")  # ↳ Nếu không phải dict -> báo lỗi rõ ràng.
    return cfg                                     # ↳ Trả dict để phần còn lại của chương trình dùng.


def save_config(cfg: dict, path: str | Path) -> None:
    # ↳ Ghi lại config (vd sau khi đã override) ra file để lưu vết thí nghiệm.
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # ↳ Tạo thư mục cha nếu chưa có (parents=True: tạo cả cây).
    with open(path, "w", encoding="utf-8") as f:          # ↳ Mở file để ghi.
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        # ↳ sort_keys=False: giữ nguyên thứ tự key như trong dict (dễ đọc theo ý mình).
        # ↳ allow_unicode=True: ghi thẳng tiếng Việt thay vì mã \uXXXX.


def apply_overrides(cfg: dict, pairs: list[str]) -> dict:
    """--set a.b=c  ->  cfg['a']['b'] = c (tự đoán kiểu int/float/bool)."""
    # ↳ Cho phép đổi 1 giá trị lồng sâu từ dòng lệnh, vd: --set train.lr=0.001
    for pair in pairs or []:                    # ↳ Duyệt từng chuỗi "key=value"; `pairs or []` = nếu None thì coi như rỗng.
        key, _, raw = pair.partition("=")       # ↳ Cắt chuỗi tại dấu "=" đầu tiên: trái = key, phải = raw (giá trị chuỗi).
        if not _:                               # ↳ `_` là dấu "=" tách được; rỗng nghĩa là không có "=" -> sai cú pháp.
            raise ValueError(f"Override '{pair}' must look like key.sub=value")
        node = cfg                              # ↳ Bắt đầu đi từ gốc dict.
        parts = key.strip().split(".")          # ↳ Tách key theo dấu chấm: "train.lr" -> ["train", "lr"].
        for p in parts[:-1]:                    # ↳ Đi qua mọi phần TRỪ phần cuối để lặn xuống dict con.
            node = node.setdefault(p, {})       # ↳ setdefault: nếu chưa có nhánh `p` thì tạo dict rỗng rồi đi vào.
        node[parts[-1]] = _coerce(raw.strip())  # ↳ Gán giá trị vào key cuối, sau khi đoán kiểu đúng (int/float/bool...).
    return cfg                                  # ↳ Trả lại cfg đã bị sửa tại chỗ.


def _coerce(v: str):
    # ↳ Chuyển chuỗi từ dòng lệnh thành đúng kiểu Python. Dấu "_" ở đầu tên = hàm nội bộ.
    # Đoán kiểu bằng yaml: '0.01'->float, 'true'->bool, '[[4,1],[4,8]]'->list, còn lại giữ str.
    try:
        out = yaml.safe_load(v)             # ↳ Mượn parser của yaml để đoán kiểu (yaml hiểu số/bool/list).
    except yaml.YAMLError:
        return v                            # ↳ Nếu yaml không parse nổi -> giữ nguyên chuỗi.
    if out is None and v.strip() not in ("null", "~", ""):
        return v                            # ↳ yaml trả None nhưng người dùng KHÔNG cố ý ghi null -> giữ chuỗi gốc.
    # yaml KHÔNG nhận '1e-4' là số (đòi '1.0e-4') -> thử float cho chuỗi trông-như-số
    if isinstance(out, str) and any(ch.isdigit() for ch in out):
        # ↳ Trường hợp '1e-4': yaml để nguyên chuỗi; nếu trong chuỗi có chữ số thì thử ép float.
        try:
            return float(out)               # ↳ '1e-4' -> 0.0001 (rất hay dùng cho learning rate).
        except ValueError:
            return out                      # ↳ Không ép được (vd 'resnet50') -> giữ chuỗi.
    return out                              # ↳ Các trường hợp còn lại: trả kết quả yaml đoán được.
