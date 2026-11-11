"""Offline test of lowspd_log.py with a fake bus: decoding, rate limit, alarm edges, and that nothing is sent."""
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
import lowspd_log as L  # noqa: E402


def frame(aid, data):
    return SimpleNamespace(arbitration_id=aid, data=bytes(data))


def low(volt_dV, drv_C, mot_C, status, bus_mA):
    return [*volt_dV.to_bytes(2, "big"), *(drv_C & 0xFFFF).to_bytes(2, "big"), mot_C & 0xFF, status,
            *bus_mA.to_bytes(2, "big")]


class FakeBus:
    def __init__(self, frames):
        self.frames, self.sent = list(frames), []

    def recv(self, timeout=None):
        return self.frames.pop(0) if self.frames else None

    def send(self, msg):                      # must never be called
        self.sent.append(msg)


class Clock:
    def __init__(self, dt):
        self.t, self.dt = 100.0, dt

    def __call__(self):
        self.t += self.dt
        return self.t


def test_decode_low():
    v = L.decode_low(low(245, 41, 36, 0b0101_0000, 1234))
    assert abs(v["volt_V"] - 24.5) < 1e-9 and v["driver_C"] == 41 and v["motor_C"] == 36
    assert v["collision"] == 1 and v["enabled"] == 1 and v["stall"] == 0 and abs(v["bus_A"] - 1.234) < 1e-9
    assert L.decode_low(low(0, -5, -3, 0, 0))["driver_C"] == -5
    assert L.decode_low(low(0, -5, -3, 0, 0))["motor_C"] == -3


def test_decode_status():
    assert L.decode_status([1, 0, 1, 0, 1, 7, 0x01, 0x02]) == (1, 0, 1, 0, 1, 7, 0x0102)


def test_run_listen_only(tmp_path):
    ok = 0b0100_0000                                                       # enabled, no alarm
    frames = []
    for k in range(40):                                                    # 40 frames of j2 at 1 ms spacing
        frames.append(frame(0x262, low(245, 40 + k // 20, 35, ok, 0)))
    frames.append(frame(0x262, low(245, 42, 35, ok | 0x10, 0)))           # collision bit set -> always written
    frames.append(frame(0x262, low(245, 42, 35, ok, 0)))                  # cleared -> always written
    frames += [frame(0x2A1, [1, 0, 1, 0, 1, 0, 0, 0])] * 3                 # unchanged status -> one row
    frames.append(frame(0x2A1, [1, 0, 1, 0, 0, 0, 0, 4]))
    frames.append(frame(0x123, [0] * 8))                                   # unrelated id ignored
    bus = FakeBus(frames)
    clk = Clock(0.001)
    out = L.main(["can_right", "--out", str(tmp_path / "o"), "--hz", "100", "--duration-s", "0.2"], bus=bus,
                 clock=clk, wall=lambda: 1.7e9)
    assert bus.sent == []
    rows = list(csv.DictReader(open(out / "lowspd.csv")))
    assert all(r["joint"] == "2" for r in rows)
    assert 2 < len(rows) < 15                                              # 100 Hz cap over ~45 ms + 2 edges
    assert sum(r["collision"] == "1" for r in rows) == 1
    st = list(csv.DictReader(open(out / "status.csv")))
    assert len(st) == 2 and st[1]["err_code"] == "4"
    meta = json.loads((out / "meta.json").read_text())
    assert meta["listen_only"] and meta["temp_range_C"]["j2"]["driver"] == [40, 42]
    assert [a[2] for a in meta["alarm_changes"]] == [["collision"], []]
