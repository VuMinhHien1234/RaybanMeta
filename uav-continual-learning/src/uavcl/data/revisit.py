"""Stream BAY LẶP LẠI — drone quay lại cùng khu vực dưới điều kiện đã đổi.

Vì sao phải có file này (xem `BAI_TOAN_VA_MUC_TIEU_2026-08-04.md` §10):

Stream trôi của D10 tăng ĐƠN ĐIỆU 0% -> 100%, điều kiện cũ **không bao giờ quay lại**. Ở đó
quên điều kiện cũ gần như miễn phí — đó chính là lý do arm λ=0,99 thắng λ=1 tới 3,91 điểm ở
mức trôi 100%.

Nhưng bài toán thật thì drone **bay lại cùng chỗ**, và mùa/giờ đều **tuần hoàn**. Quên điều
kiện mùa đông vào tháng 6 nghĩa là tháng 12 phải học lại từ đầu. Con số +3,91 điểm KHÔNG
chuyển sang được và có thể đảo dấu.

Khác biệt cốt lõi so với `drift.py`:

    drift.py    : mức trôi = hàm ĐƠN ĐIỆU của chỉ số task      -> không lặp
    revisit.py  : mức trôi = hàm TUẦN HOÀN của chỉ số chuyến   -> CÓ lặp

Và mỗi chuyến bay được gắn `mode_that` — nhãn điều kiện thật, **chỉ dùng để chấm điểm**,
model không bao giờ thấy. Nhờ nó mới đo được thước đo trung tâm của dự án:

    Lợi ích khi quay lại = Acc(gặp lại điều kiện X) − Acc(lần đầu gặp X)

Nếu con số này ≈ 0 thì hệ đang thích nghi lại từ đầu mỗi lần, tức KHÔNG ghi nhớ được điều
kiện — và mục tiêu O3 thất bại dù O1/O2 có tốt đến đâu.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

CHE_DO = ("tuan_hoan", "troi_dan", "hon_hop")


@dataclass
class ChuyenBay:
    """Một chuyến bay: đi qua TOÀN BỘ khu vực, dưới MỘT điều kiện."""
    chi_so: int          # chuyến thứ mấy
    muc_troi: float      # mức trôi [0, 1] — model không thấy
    mode_that: int       # nhãn chế độ điều kiện — CHỈ để chấm điểm
    lan_gap_mode: int    # đây là lần thứ mấy gặp chế độ này (1 = lần đầu)


def lich_bay(n_chuyen: int, che_do: str = "tuan_hoan", chu_ky: int = 4,
             severity: float = 1.0, troi_dai_han: float = 0.0,
             n_mode: int = 4) -> List[ChuyenBay]:
    """Sinh lịch điều kiện cho `n_chuyen` chuyến bay.

    tuan_hoan : hình sin, chu kỳ `chu_ky` chuyến — mùa quay vòng.
                Chuyến `chu_ky+1` lặp lại điều kiện chuyến 1  <- KỊCH BẢN CHÍNH
    troi_dan  : tăng đều 0 -> severity, KHÔNG lặp — giống D10, để nối lại kết quả cũ
    hon_hop   : trôi dài hạn CỘNG dao động mùa — thực tế nhất (khí hậu ấm dần + mùa quay vòng)

    `mode_that` rời rạc hoá mức trôi thành `n_mode` khoảng, để đếm "lần thứ mấy gặp".
    """
    if che_do not in CHE_DO:
        raise ValueError(f"che_do phải thuộc {CHE_DO} (nhận {che_do!r})")
    if n_chuyen < 1:
        raise ValueError("n_chuyen phải ≥ 1")
    if chu_ky < 2:
        raise ValueError("chu_ky phải ≥ 2 thì mới có chuyện quay lại")

    ra: List[ChuyenBay] = []
    dem_dot: dict[int, int] = {}      # số ĐỢT (không phải số chuyến) đã gặp mỗi chế độ
    mode_truoc: int | None = None
    for t in range(n_chuyen):
        if che_do == "troi_dan":
            m = severity * t / max(n_chuyen - 1, 1)
        else:
            # sin bắt đầu ở 0, lên 1 ở giữa chu kỳ, về 0 cuối chu kỳ -> điều kiện QUAY LẠI
            dao_dong = 0.5 * (1.0 - math.cos(2.0 * math.pi * t / chu_ky))
            m = severity * dao_dong
            if che_do == "hon_hop":
                m = min(1.0, m + troi_dai_han * t / max(n_chuyen - 1, 1))

        # BẮT BUỘC làm tròn: cos(π/2) trả 6,1e-17 chứ không phải 0, nên chuyến 1 và chuyến 3
        # cùng ở mức 50% lại ra 0.49999999999999994 và 0.5000000000000001 -> rơi vào HAI chế
        # độ khác nhau. Cùng một điều kiện vật lý mà bị coi là hai điều kiện thì O3 vô nghĩa.
        m = round(float(m), 6)

        # Rời rạc hoá -> "chế độ điều kiện". round (không phải floor) để mức 1.0 không rơi
        # ra ngoài dải, và để hai chuyến mức gần nhau về cùng một chế độ.
        mode = int(round(m * (n_mode - 1)))

        # ⭐ Đếm theo ĐỢT, không theo số lần xuất hiện. Với `troi_dan`, các chuyến liên tiếp
        # có mức gần nhau sẽ rơi cùng một ô rời rạc — nhưng đó KHÔNG phải "quay lại", điều
        # kiện có rời đi đâu mà quay lại. Quay lại thật = đã rời khỏi chế độ rồi trở về.
        # Không phân biệt chỗ này thì `troi_dan` sẽ báo có quay lại một cách giả tạo, và
        # thước đo O3 sẽ đo một hiện tượng không tồn tại.
        if mode != mode_truoc:
            dem_dot[mode] = dem_dot.get(mode, 0) + 1
        mode_truoc = mode
        ra.append(ChuyenBay(chi_so=t, muc_troi=m, mode_that=mode,
                            lan_gap_mode=dem_dot[mode]))
    return ra


def lich_tu_cfg(n_chuyen: int, drift_cfg: dict) -> List[ChuyenBay]:
    """A3 (KE_HOACH_SUA 2026-08-04) — MỘT chỗ đọc tham số duy nhất.

    Vì sao: trước đây `run_g1.py` và `loaders.py` mỗi nơi tự đọc drift_cfg và tự điền
    mặc định. Hiện khớp nhau, nhưng sửa mặc định một nơi là hai lịch lệch NGẦM — model
    train trên lịch này, chấm điểm trên lịch kia, và không có lỗi nào được ném ra.
    Mọi call site bắt buộc đi qua hàm này.
    """
    return lich_bay(n_chuyen,
                    che_do=str(drift_cfg.get("che_do", "tuan_hoan")),
                    chu_ky=int(drift_cfg.get("chu_ky", 4)),
                    severity=float(drift_cfg.get("severity", 1.0)),
                    troi_dai_han=float(drift_cfg.get("troi_dai_han", 0.0)),
                    n_mode=int(drift_cfg.get("n_mode", 4)))


def bang_lich_bay(lich: List[ChuyenBay]) -> str:
    """Chuỗi mô tả — in ra lúc dựng loader để log có bằng chứng lịch đúng như thiết kế."""
    d = "  ".join(f"c{c.chi_so}:{c.muc_troi * 100:.0f}%/m{c.mode_that}"
                  + ("" if c.lan_gap_mode == 1 else f"(lần {c.lan_gap_mode})")
                  for c in lich)
    lap = [c for c in lich if c.lan_gap_mode > 1]
    return f"{d}\n           số chuyến LẶP LẠI chế độ cũ: {len(lap)}/{len(lich)}"


def kiem_lich(lich: List[ChuyenBay]) -> None:
    """Chặn cấu hình vô nghĩa TRƯỚC khi đốt giờ máy.

    Lịch không có chuyến nào lặp lại chế độ cũ thì `Lợi ích khi quay lại` không tính được,
    và cả thí nghiệm mất mục tiêu chính (O3).
    """
    lap = sum(1 for c in lich if c.lan_gap_mode > 1)
    if lap == 0:
        raise ValueError(
            "Lịch bay KHÔNG có chuyến nào lặp lại chế độ điều kiện cũ -> không đo được "
            "'Lợi ích khi quay lại' (mục tiêu O3). Tăng n_chuyen, giảm chu_ky, "
            "hoặc đổi che_do sang 'tuan_hoan'.")
