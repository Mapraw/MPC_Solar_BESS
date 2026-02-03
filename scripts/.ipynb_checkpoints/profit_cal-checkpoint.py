
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional, Literal, Tuple

import pandas as pd
import numpy as np


# -----------------------------
# ช่วงเวลา (Period) ตามข้อ 9.2
# -----------------------------
# Period 1: 09:00 – 16:00
# Period 2: 18:01 – 24:00 และ 00:00 – 06:00
# Period 3: 06:01 – 09:00 และ 16:01 – 18:00

Period = Literal[1, 2, 3]

P1_START = time(9, 0, 0)
P1_END   = time(16, 0, 0)

P2A_START = time(18, 1, 0)
P2A_END   = time(23, 59, 59)   # ใช้ 23:59:59 แทน 24:00
P2B_START = time(0, 0, 0)
P2B_END   = time(6, 0, 0)

P3A_START = time(6, 1, 0)
P3A_END   = time(9, 0, 0)
P3B_START = time(16, 1, 0)
P3B_END   = time(18, 0, 0)


def infer_period(start: datetime, end: datetime) -> Period:
    """
    ระบุช่วงเวลา (Period) จากเวลาเริ่ม-สิ้นสุดของอินเตอร์วอล
    หมายเหตุ: ฟังก์ชันนี้สมมติว่าอินเตอร์วอลไม่ข้ามเส้นแบ่งช่วงเวลา
    หากข้ามช่วงเวลา ควรแยกอินเตอร์วอลก่อน
    """
    if start.date() != end.date():
        # อินเตอร์วอลข้ามวัน: ไม่รองรับในตัวนี้ (ควรตัดให้ไม่ข้ามวัน)
        raise ValueError("Interval crosses midnight (start.date() != end.date()). Please split the interval by day.")

    s = start.time()
    e = end.time()

    # Helper: ตรวจว่าช่วงทั้งหมดอยู่ภายในกรอบเวลา [A, B] แบบปิด-เปิด (รวมต้น, ไม่รวมปลาย)
    def within(s_t: time, e_t: time, A: time, B: time) -> bool:
        return (s_t >= A) and (e_t <= B)

    # Period 1: 09:00–16:00
    if within(s, e, P1_START, P1_END):
        return 1

    # Period 2: 18:01–24:00 หรือ 00:00–06:00
    if within(s, e, P2A_START, time(23,59,59)) or within(s, e, P2B_START, P2B_END):
        return 2

    # Period 3: 06:01–09:00 หรือ 16:01–18:00
    if within(s, e, P3A_START, P3A_END) or within(s, e, P3B_START, P3B_END):
        return 3

    raise ValueError(
        f"Interval {start}–{end} does not fully lie within a defined Period window or crosses boundaries. "
        "Please split intervals at period edges."
    )


@dataclass
class RevenueResult:
    revenue: float
    payable_energy_kwh: float
    penalty_energy_kwh: float
    penalty_value: float
    plan_energy_kwh: float
    cap_energy_kwh: float
    period: Period


def calculate_interval_revenue(
    start: datetime,
    end: datetime,
    actual_energy_kwh: Optional[float],
    fit_rate_per_kwh: float,
    contract_power_kw: float,
    *,
    # ถ้ามีแผน (พลังงานตามคำสั่ง กฟผ.) ต่ออินเตอร์วอลอยู่แล้ว ใส่ได้
    egat_plan_energy_kwh: Optional[float] = None,
    # Factor การสั่งการช่วงที่ 2 (0–0.6), ถ้าไม่ส่ง จะใช้ 0.6 ตามกฎสูงสุด
    egat_order_factor_p2: Optional[float] = None,
    # ช่วงที่ 3 ดีฟอลต์จ่าย FiT ได้ไม่เกิน 100% ของกำลังตามสัญญา (× เวลา)
    period3_cap_factor: float = 1.0,
    # อัตราบทลงโทษเมื่อส่งน้อยกว่าแผน (ข้อ 17.1.3 และ 17.2.3)
    penalty_rate: float = 0.12,
    # ไม่ให้รายได้ติดลบ (True) เป็นดีฟอลต์ เพื่อความปลอดภัย
    allow_negative_penalty: bool = False,
    # กรณีพิเศษ 18:01–18:15 (ข้อ 17.2.4) หากค่ามิเตอร์เป็น 15 นาที
    is_1801_1815: bool = False,
    meter_kwh_15m: Optional[float] = None,
    # ระบุ Period เองได้ ถ้าไม่ได้ให้ infer
    period_hint: Optional[Period] = None,
) -> RevenueResult:
    """
    คำนวณรายได้รายอินเตอร์วอลตามกฎในข้อความที่ให้มา (ข้อ 9.2 และ 17.x)

    พารามิเตอร์สำคัญ:
    - actual_energy_kwh: พลังงานจริง (kWh) ที่จ่ายเข้าระบบในช่วงเวลา
      * ถ้าเป็นช่วง 18:01–18:15 และมีเพียงค่ามิเตอร์ 15 นาที ให้ส่งผ่าน meter_kwh_15m และตั้ง is_1801_1815=True
        ตัวฟังก์ชันจะคิด actual_energy_kwh = meter_kwh_15m * (14/15)
    - egat_plan_energy_kwh: แผนการรับซื้อไฟฟ้า (พลังงาน) ต่อช่วง หากไม่ส่งมา
      * ช่วงที่ 1: จะตั้ง = 100% * contract_power_kw * duration_hours
      * ช่วงที่ 2: จะตั้ง = min(egat_order_factor_p2, 0.6) * contract_power_kw * duration_hours (ดีฟอลต์ egat_order_factor_p2 = 0.6)
      * ช่วงที่ 3: ใช้เป็นเพดานการจ่าย = period3_cap_factor * contract_power_kw * duration_hours (ไม่มีบทลงโทษ)
    """

    if period_hint is not None:
        period = period_hint
    else:
        period = infer_period(start, end)

    # คำนวณชั่วโมงของช่วงเวลา
    duration_hours = (end - start).total_seconds() / 3600.0
    if duration_hours <= 0:
        raise ValueError("Interval duration must be positive.")

    # จัดการกรณีพิเศษ 18:01–18:15
    if is_1801_1815:
        if meter_kwh_15m is None and actual_energy_kwh is None:
            raise ValueError("For 18:01–18:15, please provide meter_kwh_15m (15-min read) or actual_energy_kwh.")
        if meter_kwh_15m is not None and actual_energy_kwh is None:
            actual_energy_kwh = meter_kwh_15m * (14.0 / 15.0)

    if actual_energy_kwh is None:
        raise ValueError("actual_energy_kwh is required unless meter_kwh_15m provided for 18:01–18:15.")

    # พลังงานเพดานตามสัญญาต่อช่วง (100% ของสัญญา × เวลา)
    contract_energy_cap_kwh = contract_power_kw * duration_hours

    # สร้างแผน (หรือเพดาน) ต่อ Period
    if period in (1, 2):
        if egat_plan_energy_kwh is None:
            if period == 1:
                # แผน = 100% ของสัญญา
                egat_plan_energy_kwh = contract_energy_cap_kwh
            else:
                # ช่วงที่ 2: แผนตามคำสั่ง กฟผ. สูงสุด 60% ของสัญญา
                factor = 0.6 if egat_order_factor_p2 is None else min(max(egat_order_factor_p2, 0.0), 0.6)
                egat_plan_energy_kwh = contract_power_kw * factor * duration_hours

        plan_energy_kwh = float(egat_plan_energy_kwh)
        cap_energy_kwh = plan_energy_kwh  # ใช้เป็นเพดานการจ่าย (ไม่จ่ายเกินแผน)
    else:
        # Period 3: ไม่มีบทลงโทษ ส่งเท่าไรจ่าย FiT ได้ไม่เกิน 100% (หรือ period3_cap_factor) ของสัญญา
        plan_energy_kwh = period3_cap_factor * contract_energy_cap_kwh
        cap_energy_kwh = plan_energy_kwh

    # ป้องกันค่าติดลบ
    actual = max(0.0, float(actual_energy_kwh))
    plan = max(0.0, float(plan_energy_kwh))
    cap = max(0.0, float(cap_energy_kwh))

    # คำนวณรายได้
    if period in (1, 2):
        if actual >= plan:
            # จ่ายไม่เกินแผน; ส่วนเกินไม่ได้รับเงิน
            payable = plan
            penalty_energy = 0.0
            penalty_value = 0.0
            revenue = fit_rate_per_kwh * payable
        else:
            # จ่ายตามจริง แล้วหักบทลงโทษ 12% FiT × (แผน − จริง)
            payable = actual
            penalty_energy = plan - actual
            penalty_value = penalty_rate * fit_rate_per_kwh * penalty_energy
            revenue = fit_rate_per_kwh * payable - penalty_value
            if not allow_negative_penalty:
                revenue = max(revenue, 0.0)
    else:
        # Period 3: ไม่มีบทลงโทษ จ่ายที่ FiT แต่ไม่เกินเพดาน
        payable = min(actual, cap)
        penalty_energy = 0.0
        penalty_value = 0.0
        revenue = fit_rate_per_kwh * payable

    return RevenueResult(
        revenue=revenue,
        payable_energy_kwh=payable,
        penalty_energy_kwh=penalty_energy,
        penalty_value=penalty_value,
        plan_energy_kwh=plan,
        cap_energy_kwh=cap,
        period=period,
    )


def check_interval_not_crossing_period(start: datetime, end: datetime) -> None:
    """
    ช่วยตรวจสอบว่าอินเตอร์วอลไม่ข้ามเส้นแบ่งช่วงเวลา
    (แนะนำให้เตรียมข้อมูลให้สอดคล้อง เช่น 15 นาที/60 นาที ที่ตัดตามกรอบช่วง)
    """
    _ = infer_period(start, end)  # จะ throw error หากไม่เข้าเกณฑ์


def calculate_revenue_dataframe(
    df: pd.DataFrame,
    fit_rate_per_kwh: float,
    contract_power_kw: float,
    *,
    start_col: str = "start",
    end_col: str = "end",
    actual_kwh_col: str = "actual_kwh",
    egat_plan_kwh_col: Optional[str] = "egat_plan_kwh",
    egat_order_factor_p2_col: Optional[str] = "egat_order_factor_p2",
    period_col: Optional[str] = None,
    is_1801_1815_col: Optional[str] = "is_1801_1815",
    meter_kwh_15m_col: Optional[str] = "meter_kwh_15m",
    period3_cap_factor: float = 1.0,
    penalty_rate: float = 0.12,
    allow_negative_penalty: bool = False,
) -> Tuple[pd.DataFrame, float]:
    """
    คำนวณรายได้ทั้ง DataFrame (รายแถว = 1 อินเตอร์วอล)
    คืนค่า (df_with_results, total_revenue)

    ข้อกำหนด:
    - df ต้องมีคอลัมน์ start/end เป็น datetime และ actual_kwh
    - หากอินเตอร์วอลข้ามช่วงเวลา ควรแยกก่อน (ฟังก์ชันนี้ตรวจและจะ error)
    - สามารถส่งคอลัมน์ period เอง (1/2/3) หรือให้ระบบ infer
    - สำหรับช่วง 18:01–18:15 หากจะใช้ตัวคูณ 14/15 ให้ระบุ is_1801_1815=True และให้ meter_kwh_15m
    """
    df = df.copy()

    # แปลงเป็น datetime หากยังไม่ใช่
    df[start_col] = pd.to_datetime(df[start_col])
    df[end_col] = pd.to_datetime(df[end_col])

    # ตรวจสอบไม่ให้ข้ามช่วงเวลา
    for i, row in df.iterrows():
        check_interval_not_crossing_period(row[start_col], row[end_col])

    # เตรียมคอลัมน์ผลลัพธ์
    df["period"] = np.nan
    df["plan_energy_kwh"] = np.nan
    df["cap_energy_kwh"] = np.nan
    df["payable_energy_kwh"] = np.nan
    df["penalty_energy_kwh"] = np.nan
    df["penalty_value"] = np.nan
    df["revenue"] = np.nan

    for i, row in df.iterrows():
        start = row[start_col]
        end = row[end_col]
        actual = row.get(actual_kwh_col, None)
        period_hint = None
        if period_col and pd.notna(row.get(period_col, np.nan)):
            # รับ period ที่ผู้ใช้ระบุมา
            period_hint = int(row[period_col])

        egat_plan_val = None
        if egat_plan_kwh_col and egat_plan_kwh_col in df.columns and pd.notna(row.get(egat_plan_kwh_col, np.nan)):
            egat_plan_val = float(row[egat_plan_kwh_col])

        egat_factor_val = None
        if egat_order_factor_p2_col and egat_order_factor_p2_col in df.columns and pd.notna(row.get(egat_order_factor_p2_col, np.nan)):
            egat_factor_val = float(row[egat_order_factor_p2_col])

        is_1801 = False
        if is_1801_1815_col and is_1801_1815_col in df.columns and bool(row.get(is_1801_1815_col, False)):
            is_1801 = True

        meter_15m = None
        if meter_kwh_15m_col and meter_kwh_15m_col in df.columns and pd.notna(row.get(meter_kwh_15m_col, np.nan)):
            meter_15m = float(row[meter_kwh_15m_col])

        res = calculate_interval_revenue(
            start=start,
            end=end,
            actual_energy_kwh=None if pd.isna(actual) else float(actual),
            fit_rate_per_kwh=fit_rate_per_kwh,
            contract_power_kw=contract_power_kw,
            egat_plan_energy_kwh=egat_plan_val,
            egat_order_factor_p2=egat_factor_val,
            period3_cap_factor=period3_cap_factor,
            penalty_rate=penalty_rate,
            allow_negative_penalty=allow_negative_penalty,
            is_1801_1815=is_1801,
            meter_kwh_15m=meter_15m,
            period_hint=period_hint,
        )

        df.at[i, "period"] = res.period
        df.at[i, "plan_energy_kwh"] = res.plan_energy_kwh
        df.at[i, "cap_energy_kwh"] = res.cap_energy_kwh
        df.at[i, "payable_energy_kwh"] = res.payable_energy_kwh
        df.at[i, "penalty_energy_kwh"] = res.penalty_energy_kwh
        df.at[i, "penalty_value"] = res.penalty_value
        df.at[i, "revenue"] = res.revenue

    total_revenue = float(df["revenue"].sum())
    return df, total_revenue
``
