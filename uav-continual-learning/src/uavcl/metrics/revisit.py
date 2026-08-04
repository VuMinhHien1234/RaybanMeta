"""Thước đo cho bài toán BAY LẶP LẠI (mục tiêu O2, O3).

Vì sao cần thước đo mới: `AAA` / `Average Accuracy` / `Forgetting` **không đo được O3**
("gặp lại điều kiện cũ thì nhận ra ngay, không học lại"). Chúng chỉ nói mô hình đúng bao
nhiêu, không nói nó có **ghi nhớ điều kiện** hay đang thích nghi lại từ đầu mỗi lần.

Bài học trực tiếp từ D10: λ hoạt động tốt (+3,91 điểm ở mức trôi 100%, 3,10σ) nhưng `AAA`
trung bình trên mọi task đã **che mất hoàn toàn** — nhìn `AAA` thì kết luận là "λ vô dụng".
Thước đo sai thì che mất kết quả đúng. Nên định nghĩa thước đo TRƯỚC khi cài cơ chế.

Quy ước: `R[i][j]` = accuracy trên chuyến bay j sau khi đã đi qua chuyến i (ma trận dưới).
"""
from __future__ import annotations

from typing import Dict, List, Sequence


def acc_hien_tai(R) -> float:
    """O1 — trung bình đường chéo: đúng bao nhiêu trên chính điều kiện ĐANG bay.

    Đây mới là con số có nghĩa cho drone. `AAA` trung bình trên mọi chuyến đã bay, mà sau
    chuyến thứ 9 thì 8/9 bộ test là điều kiện QUÁ KHỨ — thưởng cho việc nhớ lịch sử, không
    thưởng cho việc bay đúng hôm nay.
    """
    n = len(R)
    return sum(float(R[i][i]) for i in range(n)) / n if n else 0.0


def loi_ich_quay_lai(R, mode_that: Sequence[int]) -> Dict[str, float]:
    """⭐ O3 — THƯỚC ĐO TRUNG TÂM CỦA DỰ ÁN.

        Lợi ích khi quay lại = Acc(gặp lại chế độ X) − Acc(lần ĐẦU gặp chế độ X)

    Nếu tầng ghi nhớ điều kiện hoạt động, chuyến bay gặp LẠI một chế độ đã biết phải tốt hơn
    hẳn lần đầu gặp nó — vì hệ đã có sẵn phép căn chỉnh, không phải học lại.

    ≈ 0 nghĩa là hệ thích nghi lại từ đầu mỗi lần: KHÔNG ghi nhớ được điều kiện, O3 thất bại
    dù O1/O2 có tốt đến đâu. Lúc đó nên bỏ tầng trung cho gọn.

    Đo trên ĐƯỜNG CHÉO (`R[i][i]`) — hiệu năng ngay tại chuyến đó, dưới đúng điều kiện đó.
    """
    lan_dau: Dict[int, float] = {}
    chenh: List[float] = []
    theo_lan: Dict[int, List[float]] = {}
    dem_dot: Dict[int, int] = {}
    mode_truoc = None
    for i, m in enumerate(mode_that):
        m = int(m)
        acc = float(R[i][i])
        # Chỉ tính là ĐỢT MỚI khi chế độ đổi. Chuyến liên tiếp cùng chế độ là cùng một đợt —
        # điều kiện chưa rời đi thì chưa gọi là "quay lại" được. Không lọc chỗ này thì stream
        # trôi đơn điệu (D10) sẽ báo có quay lại một cách giả tạo.
        if m == mode_truoc:
            continue
        mode_truoc = m
        dem_dot[m] = dem_dot.get(m, 0) + 1
        if m not in lan_dau:
            lan_dau[m] = acc
            continue
        d = acc - lan_dau[m]
        chenh.append(d)
        theo_lan.setdefault(dem_dot[m], []).append(d)

    ra = {
        "loi_ich_quay_lai": sum(chenh) / len(chenh) if chenh else 0.0,
        "so_lan_quay_lai": len(chenh),
        "so_che_do": len(lan_dau),
    }
    for lan, v in sorted(theo_lan.items()):
        ra[f"loi_ich_lan_{lan}"] = sum(v) / len(v)
    return ra


def thoi_gian_hoi_phuc(acc_theo_buoc: Sequence[float], nguong: float = 0.95) -> float:
    """O2 — bao nhiêu BƯỚC để accuracy hồi về `nguong` × mức ổn định sau khi điều kiện đổi.

    Mức ổn định lấy là trung bình 20% bước cuối. Trả về `inf` nếu không bao giờ hồi tới ngưỡng
    — trường hợp đó có nghĩa là hệ **không** bám kịp trôi, phải báo động chứ không lấp liếm
    bằng một con số lớn.
    """
    v = [float(x) for x in acc_theo_buoc]
    if not v:
        return float("inf")
    k = max(1, len(v) // 5)
    on_dinh = sum(v[-k:]) / k
    muc = nguong * on_dinh
    for i, x in enumerate(v):
        if x >= muc:
            return float(i)
    return float("inf")


def tom_tat_revisit(R, mode_that: Sequence[int]) -> Dict[str, float]:
    """Gom cả ba thước đo — gọi từ run_g1.py sau khi chạy xong."""
    ra = {"acc_dieu_kien_hien_tai": acc_hien_tai(R)}
    ra.update(loi_ich_quay_lai(R, mode_that))
    # Giữ nguyên acc trên chuyến ĐẦU và chuyến CUỐI để nhìn xu thế thô.
    if len(R):
        ra["acc_chuyen_dau"] = float(R[0][0])
        ra["acc_chuyen_cuoi"] = float(R[-1][-1])
    return ra


def in_bao_cao(tt: Dict[str, float]) -> str:
    """Chuỗi in ra log — để đọc một lần là biết O3 có đạt không."""
    li = tt.get("loi_ich_quay_lai", 0.0)
    dong = [
        f"  Acc điều kiện hiện tại : {tt.get('acc_dieu_kien_hien_tai', 0):.4f}   (O1)",
        f"  Lợi ích khi quay lại   : {li:+.4f}   (O3) ⭐ "
        f"trên {int(tt.get('so_lan_quay_lai', 0))} lần gặp lại / {int(tt.get('so_che_do', 0))} chế độ",
    ]
    if tt.get("so_lan_quay_lai", 0) == 0:
        dong.append("  ⚠️ KHÔNG có chuyến nào lặp lại chế độ cũ — lịch bay sai, O3 không đo được")
    elif li < 0.005:
        dong.append("  ⚠️ Lợi ích ≈ 0 — hệ đang thích nghi LẠI TỪ ĐẦU mỗi lần, KHÔNG ghi nhớ điều kiện")
    return "\n".join(dong)
