from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from geotrace.config import G_TO_MS2
from geotrace.live_logs import IMU_ACCEL, IMU_GYRO, imu_to_motion_samples
from geotrace.rfid_logs import find_complete_laps, read_real_imu_rows, read_rfid_anchors


def test_real_imu_si_units_reuse_the_vehicle_logger_conversion(tmp_path: Path) -> None:
    path = tmp_path / "imu.csv"
    path.write_text(
        "device_id,time,imu_ms,ax,ay,az,gx,gy,gz,mx,my,mz,pitch,roll,heading,"
        "heading_valid,quat_w,quat_x,quat_y,quat_z,temperature\n"
        f"911,1000,0,0,0,{G_TO_MS2},0,0,{math.pi / 2},0,0,0,0,0,0,true,1,0,0,0,20\n"
        "broken row\n"
        f"911,1020,20,0,0,{G_TO_MS2},0,0,0,0,0,0,0,0,0,true,1,0,0,0,20\n",
        encoding="utf-8",
    )
    rows = read_real_imu_rows(path)
    assert rows.shape == (2, 11)
    assert rows[0, IMU_ACCEL] == pytest.approx((0.0, 0.0, 1.0))
    assert rows[0, IMU_GYRO] == pytest.approx((0.0, 0.0, 90.0))
    sample = imu_to_motion_samples(rows[:1], 1000.0, rate_hz=0.0)[0]
    assert sample.user_acceleration_g == pytest.approx((0.0, 0.0, 0.0))
    assert sample.rotation_rate[2] == pytest.approx(math.pi / 2)


def test_rfid_laps_are_only_complete_bounded_cycles(tmp_path: Path) -> None:
    path = tmp_path / "anchors.csv"
    header = (
        "device_id,host_time_ms,anchor_type,anchor_id,anchor_ts_ms,"
        "imu_sample_ts_ms,dt_residual_ms,clamped,heading,lat,lon\n"
    )
    rows = []
    ids = ["1", *map(str, range(2, 17)), "1", *map(str, range(2, 17)), "1"]
    for index, anchor_id in enumerate(ids):
        rows.append(
            f"911,{index * 1000},rfid,{anchor_id},{index * 1000},"
            f"{index * 1000},0,false,90,59.9,30.3"
        )
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    anchors = read_rfid_anchors(path)
    laps = find_complete_laps(anchors)
    assert len(laps) == 2
    assert laps[0].anchor_count == 17
    assert laps[0].duration_s == 16.0


# Imported at the bottom to keep the production module's import smoke visible
# even when pytest rewrites this test module.
import pytest
