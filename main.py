# 2636B Measurement Software — White UI (v4.2)
# 变更（相对 v4/v4.1）：
# - 启动仅自动连接“型号/IDN 含 2636B”的设备；没找到就不连接（保留未连接/模拟器）。
# - 扫描提速：并发查询 + 超短超时（~120–150ms），仅在需要时查询 *IDN?*。
# - 默认工作模式改为“电压源”。
# - 剩余时间 ETA 改为：未扫描点数 / (已扫描点数 / 已用时间)，实时根据扫描速率刷新。
from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import queue
import random
import statistics
import threading
import time
import tkinter as tk
from dataclasses import dataclass, asdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pyvisa
    from pyvisa import VisaIOError
except Exception:  # pragma: no cover
    pyvisa = None  # type: ignore
    VisaIOError = Exception  # type: ignore


CFG_PATH = Path.home() / ".2636b_gui_config.json"

def fmt_e(val: float, digits: int = 3) -> str:
    try:
        return f"{float(val):.{digits}e}"
    except Exception:
        return str(val)

def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


# ----------------------------- Instruments ------------------------------
class InstrumentError(RuntimeError):
    pass

@dataclass
class VisaResourceInfo:
    resource: str
    idn: str = ""
    alias: Optional[str] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    serial: Optional[str] = None
    firmware: Optional[str] = None

    def display_name(self) -> str:
        base = self.model or self.alias or self.resource
        details = []
        if self.serial:
            details.append(f"SN {self.serial}")
        if self.idn:
            details.append(self.idn)
        info = " | ".join(details)
        return f"{base} ({self.resource})" if not info else f"{base} ({info})"


# ---------- VISA discovery (fast, concurrent) ----------
def _query_idn_for(name: str, timeout_s: float = 0.15) -> VisaResourceInfo:
    """Open a resource briefly and try to read *IDN?* quickly."""
    info = VisaResourceInfo(resource=name)
    if pyvisa is None:
        return info
    try:
        rm = pyvisa.ResourceManager()  # type: ignore
    except Exception:
        return info
    try:
        h = None
        try:
            h = rm.open_resource(name)
            try:
                h.timeout = int(timeout_s * 1000)
            except Exception:
                pass
            try:
                info.alias = rm.resource_info(name).alias or None
            except Exception:
                pass
            try:
                idn = str(h.query("*IDN?")).strip()
            except Exception:
                idn = ""
            info.idn = idn
            if idn:
                parts = [p.strip() for p in idn.split(",")]
                if parts: info.manufacturer = parts[0]
                if len(parts) > 1: info.model = parts[1]
                if len(parts) > 2: info.serial = parts[2]
                if len(parts) > 3: info.firmware = parts[3]
        finally:
            if h is not None:
                try: h.close()
                except Exception: pass
    finally:
        try: rm.close()
        except Exception: pass
    return info

def discover_visa_resources_fast(timeout_s: float = 0.12, with_idn: bool = False, max_workers: int = 10) -> List[VisaResourceInfo]:
    """Fast discovery. If with_idn=True, query *IDN?* concurrently with short timeouts."""
    if pyvisa is None:
        return []
    try:
        rm = pyvisa.ResourceManager()  # type: ignore
    except Exception:
        return []
    try:
        names = list(rm.list_resources())
    except Exception:
        names = []
    finally:
        try: rm.close()
        except Exception: pass

    if not with_idn:
        # Only basic stubs (fastest)
        return [VisaResourceInfo(resource=n) for n in names]

    # Concurrent IDN queries
    out: List[VisaResourceInfo] = []
    qin: "queue.Queue[str]" = queue.Queue()
    qout: "queue.Queue[VisaResourceInfo]" = queue.Queue()
    for n in names: qin.put(n)
    stop = object()

    def worker():
        while True:
            try:
                n = qin.get_nowait()
            except Exception:
                break
            try:
                qout.put(_query_idn_for(n, timeout_s=timeout_s))
            finally:
                qin.task_done()
        qout.put(stop)  # signal

    nthreads = max(1, min(max_workers, len(names)))
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(nthreads)]
    for t in threads: t.start()
    alive = len(threads)
    while alive > 0:
        item = qout.get()
        if item is stop:
            alive -= 1
        else:
            out.append(item)  # collect
    return out

def discover_first_2636b(timeout_s: float = 0.15, max_workers: int = 10) -> Optional[VisaResourceInfo]:
    """Find first resource whose IDN/model contains '2636B' quickly; None if not found."""
    if pyvisa is None:
        return None
    try:
        rm = pyvisa.ResourceManager()  # type: ignore
        names = list(rm.list_resources())
    except Exception:
        names = []
    finally:
        try: rm.close()
        except Exception: pass
    if not names:
        return None

    result_lock = threading.Lock()
    found: List[VisaResourceInfo] = []

    def worker(name: str):
        if found:  # early check
            return
        info = _query_idn_for(name, timeout_s=timeout_s)
        label = " ".join([info.resource or "", info.model or "", info.idn or ""]).upper()
        if "2636B" in label:
            with result_lock:
                if not found:
                    found.append(info)

    threads = [threading.Thread(target=worker, args=(n,), daemon=True) for n in names]
    for t in threads: t.start()
    # Wait up to ~timeout_s*2 for a winner, but not to block forever
    end = time.perf_counter() + max(timeout_s * 2, 0.3)
    while time.perf_counter() < end and not found and any(t.is_alive() for t in threads):
        time.sleep(0.02)
    # If still nothing, join briefly and return None
    return found[0] if found else None


# ----------------------------- Simulation ------------------------------
@dataclass
class MeasurementPoint:
    index: int
    timestamp: _dt.datetime
    voltage: float
    current: float

    @property
    def resistance(self) -> float:
        try:
            return self.voltage / self.current
        except ZeroDivisionError:
            return float("inf")

    @property
    def power(self) -> float:
        return self.voltage * self.current

@dataclass
class InstrumentSettings:
    channel: str = "Channel A"
    mode: str = "电压源"           # 默认改为“电压源”
    autorange: bool = True
    nplc: float = 1.0
    trigger_delay_ms: float = 10.0
    compliance_voltage: float = 10.0  # 电流源：合规电压
    compliance_current: float = 0.01  # 电压源：钳位电流
    start_level: float = 0.0
    stop_level: float = 1.0
    step: float = 0.1
    sweep_points: int = 11
    output_enable: bool = True

class InstrumentSimulator:
    def __init__(self) -> None:
        self.settings = InstrumentSettings()
        self._is_connected = False
        self._rng = random.Random(2636)

    def connect(self) -> None:
        time.sleep(0.02)
        self._is_connected = True

    def disconnect(self) -> None:
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def prepare_measurement(self, _setpoints: Sequence[float]) -> None:
        pass

    def generate_point(self, index: int, level: Optional[float] = None) -> MeasurementPoint:
        s = self.settings
        level = float(level or 0.0)
        angle = (abs(level) + index * 0.02) * math.tau * 0.05
        nv = self._rng.uniform(-0.02, 0.02)
        ni = self._rng.uniform(-0.0001, 0.0001)
        if s.mode == "电流源":
            target_i = level
            base_v = 5.0 * math.sin(angle) + target_i * 200.0
            v = base_v
            i = target_i
            clamp_v = abs(s.compliance_voltage)
            if clamp_v > 0 and abs(v) > clamp_v:
                v = math.copysign(clamp_v, v)
                i = (v - 5.0 * math.sin(angle)) / 200.0
        else:
            target_v = level
            base_i = 0.01 * math.cos(angle) + target_v / 200.0
            v = target_v
            i = base_i
            clamp_i = abs(s.compliance_current)
            if clamp_i > 0 and abs(i) > clamp_i:
                i = math.copysign(clamp_i, i)
                v = (i * 200.0) - 5.0 * math.sin(angle)
        return MeasurementPoint(index=index, timestamp=_dt.datetime.now(), voltage=v + nv, current=i + ni)

    def finalize_measurement(self) -> None: pass
    def abort_measurement(self) -> None: pass

class VisaInstrument:
    def __init__(self, info: VisaResourceInfo, *, timeout: float = 10.0) -> None:
        self.settings = InstrumentSettings()
        self.info = info
        self.timeout = timeout
        self._rm = None
        self._res = None
        self._is_connected = False
        self._channel = "smua"

    def connect(self) -> None:
        if pyvisa is None:
            raise InstrumentError("未安装 PyVISA")
        try:
            self._rm = pyvisa.ResourceManager()  # type: ignore
            self._res = self._rm.open_resource(self.info.resource)
            self._res.timeout = int(self.timeout * 1000)
            self._res.write("*CLS")
            self._res.write("format.data = format.ASCII")
            self._res.write("format.asciiprecision = 6")
            self._res.write("format.asciiexponent = 3")
            self._is_connected = True
        except Exception as exc:
            raise InstrumentError(f"连接失败: {exc}") from exc

    def disconnect(self) -> None:
        try:
            self.abort_measurement()
        finally:
            try:
                if self._res: self._res.close()
            except Exception: pass
            try:
                if self._rm: self._rm.close()
            except Exception: pass
            self._res = None; self._rm = None; self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def _write(self, cmd: str) -> None:
        if not self._res:
            raise InstrumentError("仪器未连接")
        try:
            self._res.write(cmd)
        except VisaIOError as exc:
            raise InstrumentError(f"通信失败: {exc}") from exc

    def _query(self, cmd: str) -> str:
        if not self._res:
            raise InstrumentError("仪器未连接")
        try:
            return str(self._res.query(cmd)).strip()
        except VisaIOError as exc:
            raise InstrumentError(f"通信失败: {exc}") from exc

    def _ch(self) -> str:
        return "smua" if self.settings.channel == "Channel A" else "smub"

    def prepare_measurement(self, setpoints: Sequence[float]) -> None:
        if not self._is_connected: self.connect()
        self._channel = self._ch()
        self._write("errorqueue.clear()"); self._write("display.clear()"); self._write(f"{self._channel}.reset()")
        self._write(f"{self._channel}.measure.nplc = {max(self.settings.nplc, 0.01)}")
        self._write(f"{self._channel}.source.output = {self._channel}.OUTPUT_OFF")
        if self.settings.mode == "电流源":
            self._write(f"{self._channel}.source.func = {self._channel}.OUTPUT_DCAMPS")
            self._write(f"{self._channel}.source.autorangei = {self._channel}.AUTORANGE_{'ON' if self.settings.autorange else 'OFF'}")
            self._write(f"{self._channel}.source.limitv = {abs(self.settings.compliance_voltage)}")
        else:
            self._write(f"{self._channel}.source.func = {self._channel}.OUTPUT_DCVOLTS")
            self._write(f"{self._channel}.source.autorangev = {self._channel}.AUTORANGE_{'ON' if self.settings.autorange else 'OFF'}")
            self._write(f"{self._channel}.source.limiti = {abs(self.settings.compliance_current)}")
        delay = max(self.settings.trigger_delay_ms / 1000.0, 0.0)
        self._write(f"{self._channel}.source.delay = {delay}")
        initial = setpoints[0] if setpoints else 0.0
        if self.settings.mode == "电流源":
            self._write(f"{self._channel}.source.leveli = {initial}")
        else:
            self._write(f"{self._channel}.source.levelv = {initial}")
        self._write(f"{self._channel}.source.output = {self._channel}.OUTPUT_ON")

    def generate_point(self, index: int, level: Optional[float] = None) -> MeasurementPoint:
        level = float(level or 0.0)
        if self.settings.mode == "电流源":
            self._write(f"{self._channel}.source.leveli = {level}")
        else:
            self._write(f"{self._channel}.source.levelv = {level}")
        q = ("print(string.format('%e,%e', "
             f"{self._channel}.measure.v(), {self._channel}.measure.i()))")
        resp = self._query(q)
        try:
            v_str, i_str = resp.split(","); v = float(v_str); i = float(i_str)
        except Exception as exc:
            raise InstrumentError(f"解析失败: {resp}") from exc
        return MeasurementPoint(index=index, timestamp=_dt.datetime.now(), voltage=v, current=i)

    def finalize_measurement(self) -> None:
        try: self._write(f"{self._channel}.source.output = {self._channel}.OUTPUT_OFF")
        except Exception: pass
    def abort_measurement(self) -> None:
        try: self._write(f"{self._channel}.source.output = {self._channel}.OUTPUT_OFF")
        except Exception: pass


# ------------------------------- Graph ---------------------------------
class SweepGraph(ttk.Frame):
    AXIS_FIELDS: Dict[str, Tuple[str, str]] = {
        "序号": ("index", ""),
        "时间 (s)": ("time", "s"),
        "电压 (V)": ("voltage", "V"),
        "电流 (A)": ("current", "A"),
        "电阻 (Ω)": ("resistance", "Ω"),
        "功率 (W)": ("power", "W"),
    }
    def __init__(self, master: tk.Widget, *, width: int = 880, height: int = 380) -> None:
        super().__init__(master, style="White.TFrame")
        self.canvas = tk.Canvas(self, width=width, height=height, background="#ffffff", highlightthickness=1, highlightbackground="#e0e0e0")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.margin = 50
        self.axis_choices = list(self.AXIS_FIELDS.keys())
        self._x_axis_label = "电压 (V)"; self._y_axis_label = "电流 (A)"
        self._base_x_label = self._x_axis_label; self._base_y_label = self._y_axis_label
        self._data: List[Tuple[float, float]] = []; self._raw: List[MeasurementPoint] = []
        self._x_range = (0.0, 1.0); self._y_range = (-1.0, 1.0)
        self._t0: Optional[_dt.datetime] = None
        self._log_x = False; self._log_y = False
        self._minor_div = 5
        self._transform_mode = "none"
        self.canvas.bind("<Configure>", lambda e: self.redraw())

    def set_log(self, logx: bool, logy: bool) -> None:
        self._log_x = logx; self._log_y = logy; self.redraw()

    def set_transform(self, mode: str) -> None:
        if self._transform_mode == mode:
            return
        self._transform_mode = mode
        self.set_data(self._raw, x_axis=self._base_x_label, y_axis=self._base_y_label)

    def set_data(self, points: Iterable[MeasurementPoint], *, x_axis: Optional[str] = None, y_axis: Optional[str] = None) -> None:
        if x_axis is not None:
            self._base_x_label = x_axis
        if y_axis is not None:
            self._base_y_label = y_axis
        self._x_axis_label = self._base_x_label
        self._y_axis_label = self._base_y_label
        self._raw = list(points); self._t0 = None; self._data = []

        xf = self.AXIS_FIELDS.get(self._x_axis_label, ("index", ""))[0]
        yf = self.AXIS_FIELDS.get(self._y_axis_label, ("current", "A"))[0]

        def v_from(p: MeasurementPoint, field: str) -> float:
            if field == "index": return float(p.index)
            if field == "time":
                nonlocal_t0 = getattr(self, "_t0")
                if nonlocal_t0 is None:
                    self._t0 = p.timestamp; nonlocal_t0 = self._t0
                return (p.timestamp - nonlocal_t0).total_seconds()
            return float(getattr(p, field))

        def to_log(v: float) -> float:
            eps = 1e-12; return math.log10(abs(v) + eps)

        transform = self._transform_mode
        if transform == "sclc":
            self._x_axis_label = "log10(|V|)"; self._y_axis_label = "log10(|I|)"
            for p in self._raw:
                v = float(p.voltage); i = float(p.current)
                if not (math.isfinite(v) and math.isfinite(i)): continue
                xv = to_log(v); yv = to_log(i)
                self._data.append((xv, yv))
        elif transform == "pf":
            self._x_axis_label = "sqrt(|V|)"; self._y_axis_label = "ln(|I/V|)"
            for p in self._raw:
                v = float(p.voltage); i = float(p.current)
                if not (math.isfinite(v) and math.isfinite(i)): continue
                if math.isclose(v, 0.0, abs_tol=1e-18): continue
                xv = math.sqrt(abs(v))
                ratio = abs(i / v)
                if ratio <= 0.0: continue
                yv = math.log(ratio)
                self._data.append((xv, yv))
        elif transform == "fn":
            self._x_axis_label = "1/V"; self._y_axis_label = "ln(|I/V^2|)"
            for p in self._raw:
                v = float(p.voltage); i = float(p.current)
                if not (math.isfinite(v) and math.isfinite(i)): continue
                if math.isclose(v, 0.0, abs_tol=1e-18): continue
                xv = 1.0 / v
                denom = v * v
                ratio = abs(i / denom)
                if ratio <= 0.0 or not math.isfinite(xv): continue
                yv = math.log(ratio)
                self._data.append((xv, yv))
        elif transform == "schottky":
            self._x_axis_label = "sqrt(|V|)"; self._y_axis_label = "ln(|I|)"
            for p in self._raw:
                v = float(p.voltage); i = float(p.current)
                if not (math.isfinite(v) and math.isfinite(i)): continue
                if abs(i) <= 0.0: continue
                xv = math.sqrt(abs(v))
                yv = math.log(abs(i))
                self._data.append((xv, yv))
        else:
            for p in self._raw:
                xv = v_from(p, xf); yv = v_from(p, yf)
                if not (math.isfinite(xv) and math.isfinite(yv)): continue
                if self._log_x: xv = to_log(xv)
                if self._log_y: yv = to_log(yv)
                self._data.append((xv, yv))

        if self._data:
            xs, ys = zip(*self._data)
            self._x_range = self._expand_range(min(xs), max(xs))
            self._y_range = self._expand_range(min(ys), max(ys))
        else:
            self._x_range = (0.0, 1.0); self._y_range = (-1.0, 1.0)
        self.redraw()

    def _expand_range(self, vmin: float, vmax: float) -> Tuple[float, float]:
        if not math.isfinite(vmin) or not math.isfinite(vmax): return (-1.0, 1.0)
        if math.isclose(vmin, vmax, rel_tol=1e-9, abs_tol=1e-12):
            pad = abs(vmin) * 0.05 or 1.0; return (vmin - pad, vmax + pad)
        span = vmax - vmin; pad = span * 0.1 or 1.0; return (vmin - pad, vmax + pad)

    def _nice_number(self, value: float) -> float:
        if value <= 0: return 1.0
        e = math.floor(math.log10(value)); f = value / (10 ** e)
        nf = 1.0 if f < 1.5 else 2.0 if f < 3.0 else 5.0 if f < 7.0 else 10.0
        return nf * (10 ** e)

    def _ticks(self, vmin: float, vmax: float, *, count: int = 6) -> Tuple[List[float], float]:
        if not math.isfinite(vmin) or not math.isfinite(vmax): return ([], 1.0)
        if math.isclose(vmax, vmin, rel_tol=1e-9, abs_tol=1e-12): vmax = vmin + 1.0
        span = vmax - vmin; step = self._nice_number(span / max(count - 1, 1))
        start = math.floor(vmin / step) * step; vals: List[float] = []; x = start
        while x <= vmax + step * 0.5: vals.append(x); x += step
        return vals, step

    def _fmt_tick(self, v: float) -> str: return fmt_e(v, 2)

    def redraw(self) -> None:
        c = self.canvas; c.delete("all")
        W = max(10, c.winfo_width()); H = max(10, c.winfo_height())
        m = self.margin; L, T, R, B = m, m, W - m, H - m
        c.create_rectangle(L, T, R, B, outline="#e5e5e5")
        xlab = f"log10({self._x_axis_label})" if self._log_x else self._x_axis_label
        ylab = f"log10({self._y_axis_label})" if self._log_y else self._y_axis_label
        c.create_text(W/2, H - m/3, text=xlab, fill="#333")
        c.create_text(m/3, H/2, text=ylab, angle=90, fill="#333")

        if not self._data:
            c.create_text(W/2, H/2, text="尚未开始测量", fill="#999"); return

        min_x, max_x = self._x_range; min_y, max_y = self._y_range
        sx = max(max_x - min_x, 1e-12); sy = max(max_y - min_y, 1e-12)
        x_ticks, x_step = self._ticks(min_x, max_x); y_ticks, y_step = self._ticks(min_y, max_y)

        # grid with minor ticks
        for v in x_ticks:
            x = L + ((v - min_x) / sx) * (R - L)
            c.create_line(x, T, x, B, fill="#f7f7f7")
            c.create_text(x, B + 12, text=self._fmt_tick(v), anchor="n", font=("Arial", 8), fill="#666")
            for k in range(1, 5):
                xm = v + k * (x_step / 5)
                if xm >= max_x: break
                xx = L + ((xm - min_x) / sx) * (R - L)
                c.create_line(xx, T, xx, B, fill="#fbfbfb")
        for v in y_ticks:
            y = B - ((v - min_y) / sy) * (B - T)
            c.create_line(L, y, R, y, fill="#f7f7f7")
            c.create_text(L - 6, y, text=self._fmt_tick(v), anchor="e", font=("Arial", 8), fill="#666")
            for k in range(1, 5):
                ym = v + k * (y_step / 5)
                if ym >= max_y: break
                yy = B - ((ym - min_y) / sy) * (B - T)
                c.create_line(L, yy, R, yy, fill="#fbfbfb")

        coords = []
        for xv, yv in self._data:
            x = L + ((xv - min_x) / sx) * (R - L)
            y = B - ((yv - min_y) / sy) * (B - T)
            coords.append((x, y))
        for i in range(1, len(coords)):
            x0, y0 = coords[i-1]; x1, y1 = coords[i]
            c.create_line(x0, y0, x1, y1, fill="#2979ff", width=2)
        for x, y in coords:
            c.create_oval(x-2, y-2, x+2, y+2, fill="#2979ff", outline="")


# ------------------------------- App -----------------------------------
class MeasurementApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("2636B Measurement")
        self.geometry("1280x800"); self.minsize(1180, 720)
        self.configure(background="#ffffff")

        # backend
        self.instrument: InstrumentSimulator | VisaInstrument = InstrumentSimulator()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.measurements: List[MeasurementPoint] = []

        # vars
        self.file_path_var = tk.StringVar()
        self.measure_name_var = tk.StringVar(value="默认测试")
        self.sample_name_var = tk.StringVar(value="Sample-001")
        self.operator_var = tk.StringVar(value="Operator")
        self.device_status_var = tk.StringVar(value="未连接")
        self.summary_var = tk.StringVar(value="准备就绪")
        self.resource_var = tk.StringVar(value="内置模拟器")
        self.resource_info_var = tk.StringVar(value="使用内置模拟器")
        self.snapshots_var = tk.BooleanVar(value=False)
        self.auto_points_mode_var = tk.StringVar(value="from_step")
        self._auto_mode_display_var = tk.StringVar(value="按步长算点数")
        self._auto_calc_label: Optional[ttk.Label] = None
        self._step_entry: Optional[ttk.Entry] = None
        self._points_entry: Optional[ttk.Entry] = None
        self.filter_enable_var = tk.BooleanVar(value=False)
        self.filter_window_var = tk.IntVar(value=5)
        self.cycle_count_var = tk.IntVar(value=1)
        self.cycle_interval_var = tk.DoubleVar(value=0.0)
        self.sweep_traj_var = tk.StringVar(value="单向")  # 单向 / 往返(Up/Down)
        self.sweep_start_var = tk.StringVar(value="从起点")  # 从起点 / 从终点 / 从零点

        self.progress_var = tk.DoubleVar(value=0.0)
        self.eta_var = tk.StringVar(value="—")

        self._resource_map: Dict[str, VisaResourceInfo] = {}
        self._setpoints: List[float] = []
        self._active_mode: str = "IV"
        self._current_cycle = 0

        # ETA runtime markers
        self._run_started_at: Optional[float] = None
        self._total_points_planned: int = 0

        # axis vars
        self.x_axis_var_iv = tk.StringVar(value="电压 (V)")
        self.y_axis_var_iv = tk.StringVar(value="电流 (A)")
        self.logx_var_iv = tk.BooleanVar(value=False); self.logy_var_iv = tk.BooleanVar(value=False)
        self.transform_mode_var_iv = tk.StringVar(value="none")
        self.transform_flags_iv = {
            "sclc": tk.BooleanVar(value=False),
            "pf": tk.BooleanVar(value=False),
            "fn": tk.BooleanVar(value=False),
            "schottky": tk.BooleanVar(value=False),
        }

        self.x_axis_var_vt = tk.StringVar(value="时间 (s)")
        self.y_axis_var_vt = tk.StringVar(value="电压 (V)")
        self.logx_var_vt = tk.BooleanVar(value=False); self.logy_var_vt = tk.BooleanVar(value=False)

        self.x_axis_var_it = tk.StringVar(value="时间 (s)")
        self.y_axis_var_it = tk.StringVar(value="电流 (A)")
        self.logx_var_it = tk.BooleanVar(value=False); self.logy_var_it = tk.BooleanVar(value=False)

        self.graph_iv = None; self.graph_vt = None; self.graph_it = None
        self.graph_active: Optional[SweepGraph] = None
        self.result_tree_iv = None; self.result_tree_vt = None; self.result_tree_it = None
        self.result_tree_active: Optional[ttk.Treeview] = None
        self._iv_axis_controls: Dict[str, ttk.Widget] = {}

        self._configure_styles()
        self._build_layout()
        self.after(50, self._load_config_then_autoconnect)
        self.protocol("WM_DELETE_WINDOW", self.on_exit)

    # styles
    def _configure_styles(self) -> None:
        st = ttk.Style(self); st.theme_use("clam")
        white = "#ffffff"; gray = "#333333"
        st.configure(".", background=white); st.configure("White.TFrame", background=white)
        st.configure("TLabel", background=white, foreground=gray)
        st.configure("TNotebook", background=white, borderwidth=0)
        st.configure("TNotebook.Tab", background=white, padding=(12, 6))
        st.configure("TButton", padding=8, background=white, foreground="#1a73e8", relief="raised", borderwidth=2)
        st.configure("Primary.TButton", padding=10, foreground=white, background="#1a73e8", relief="raised", borderwidth=2)
        st.configure("Danger.TButton", padding=10, foreground=white, background="#d9534f", relief="raised", borderwidth=2)
        st.map("TButton", relief=[("pressed", "sunken"), ("active", "raised")])
        st.map("Primary.TButton", relief=[("pressed", "sunken"), ("active", "raised")])
        st.map("Danger.TButton", relief=[("pressed", "sunken"), ("active", "raised")])
        st.configure("Treeview", background=white, fieldbackground=white, foreground="#222", rowheight=24, borderwidth=0)
        st.configure("Treeview.Heading", background=white, foreground="#444")

    # layout
    def _build_layout(self) -> None:
        meta = ttk.Frame(self, style="White.TFrame")
        meta.pack(fill=tk.X, padx=16, pady=(12, 6))
        ttk.Label(meta, text="测试名称").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Entry(meta, width=18, textvariable=self.measure_name_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(meta, text="样品编号").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Entry(meta, width=16, textvariable=self.sample_name_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(meta, text="操作者").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Entry(meta, width=12, textvariable=self.operator_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(meta, text="仪器状态：").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Label(meta, textvariable=self.device_status_var, font=("Microsoft YaHei", 10, "bold")).pack(side=tk.LEFT)

        body = ttk.Frame(self, style="White.TFrame"); body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 8))

        # Left (narrow)
        left = ttk.Frame(body, style="White.TFrame", width=250)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 8))
        left.pack_propagate(False)
        self._build_left_panel(left)

        # Right
        right = ttk.Frame(body, style="White.TFrame")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(right); self.notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", lambda _e: self._on_tab_changed())

        iv_tab = ttk.Frame(self.notebook, style="White.TFrame")
        self.notebook.add(iv_tab, text="I/V测量")
        self._build_iv_tab(iv_tab)

        vt_tab = ttk.Frame(self.notebook, style="White.TFrame")
        self.notebook.add(vt_tab, text="V/t测试")
        self._build_time_tab(vt_tab, mode="Vt")

        it_tab = ttk.Frame(self.notebook, style="White.TFrame")
        self.notebook.add(it_tab, text="I/t测试")
        self._build_time_tab(it_tab, mode="It")

        # status + progress
        status_area = ttk.Frame(self, style="White.TFrame")
        status_area.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0, 8))
        self.status_label = ttk.Label(status_area, textvariable=self.summary_var, anchor="w")
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        prog_container = ttk.Frame(status_area, style="White.TFrame"); prog_container.pack(side=tk.RIGHT)
        ttk.Label(prog_container, text="进度").pack(side=tk.LEFT, padx=(0, 6))
        self.progress = ttk.Progressbar(prog_container, length=260, mode="determinate", variable=self.progress_var, maximum=100.0)
        self.progress.pack(side=tk.LEFT)
        ttk.Label(prog_container, textvariable=self.eta_var).pack(side=tk.LEFT, padx=(8, 0))

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        dev = ttk.LabelFrame(parent, text="设备", style="White.TFrame")
        dev.pack(fill=tk.X, padx=6, pady=(6, 6))
        ttk.Button(dev, text="扫描设备", command=self.scan_instruments).pack(fill=tk.X, pady=(4, 2))
        self.resource_combo = ttk.Combobox(dev, textvariable=self.resource_var, state="readonly")
        self.resource_combo.pack(fill=tk.X, pady=2)
        self.resource_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_resource_selected())
        ttk.Label(dev, textvariable=self.resource_info_var, wraplength=220, foreground="#666").pack(fill=tk.X, pady=(2, 6))
        ttk.Button(dev, text="连接仪器", command=self.connect_instrument).pack(fill=tk.X, pady=2)
        ttk.Button(dev, text="断开连接", command=self.disconnect_instrument).pack(fill=tk.X, pady=2)

        files = ttk.LabelFrame(parent, text="文件", style="White.TFrame")
        files.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Label(files, text="保存目录").pack(anchor="w", padx=4, pady=(6, 0))
        ttk.Entry(files, textvariable=self.file_path_var).pack(fill=tk.X, padx=4, pady=(2, 4))
        ttk.Button(files, text="浏览...", command=self.select_directory).pack(fill=tk.X, padx=4, pady=(0, 6))
        ttk.Checkbutton(files, text="逐点快照（每点保存 1 份 CSV）", variable=self.snapshots_var).pack(anchor="w", padx=4, pady=(0, 8))

        logs = ttk.LabelFrame(parent, text="日志", style="White.TFrame")
        logs.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.log_text = tk.Text(logs, height=12, wrap=tk.WORD, background="#ffffff", foreground="#333", relief="solid", bd=1)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.log_text.configure(state=tk.DISABLED)

        action = ttk.Frame(parent, style="White.TFrame")
        action.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 8))
        self.start_btn = ttk.Button(action, text="开始测量", style="Primary.TButton", command=self.start_measurement)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.stop_btn = ttk.Button(action, text="停止", style="Danger.TButton", command=self.stop_measurement, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)

    # ----------------------- Right: IV tab (3 columns) ------------------
    def _build_iv_tab(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(0, weight=6)
        parent.rowconfigure(1, weight=2)
        parent.rowconfigure(2, weight=3)
        for c in (0, 1, 2):
            parent.columnconfigure(c, weight=1)

        # Row 0: Graph area
        top = ttk.Frame(parent, style="White.TFrame")
        top.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=8, pady=(6, 4))

        axis = ttk.Frame(top, style="White.TFrame")
        axis.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(axis, text="横轴").pack(side=tk.LEFT)
        x_combo = ttk.Combobox(axis, textvariable=self.x_axis_var_iv, state="readonly", width=14)
        x_combo.pack(side=tk.LEFT, padx=(6, 18))
        ttk.Label(axis, text="纵轴").pack(side=tk.LEFT)
        y_combo = ttk.Combobox(axis, textvariable=self.y_axis_var_iv, state="readonly", width=14)
        y_combo.pack(side=tk.LEFT, padx=(6, 18))
        logx_chk = ttk.Checkbutton(axis, text="X 取 log10", variable=self.logx_var_iv, command=self._refresh_graph)
        logx_chk.pack(side=tk.LEFT, padx=10)
        logy_chk = ttk.Checkbutton(axis, text="Y 取 log10", variable=self.logy_var_iv, command=self._refresh_graph)
        logy_chk.pack(side=tk.LEFT)

        transforms = ttk.Frame(axis, style="White.TFrame")
        transforms.pack(side=tk.LEFT, padx=(16, 0))
        ttk.Label(transforms, text="自动换算").grid(row=0, column=0, sticky="w", padx=(0, 6))
        transform_defs = [
            ("sclc", "SCLC"),
            ("pf", "PF"),
            ("fn", "FN"),
            ("schottky", "肖特基"),
        ]
        for idx, (mode, text) in enumerate(transform_defs):
            chk = ttk.Checkbutton(
                transforms,
                text=text,
                variable=self.transform_flags_iv[mode],
                command=lambda m=mode: self._toggle_transform_mode(m),
            )
            chk.grid(row=0, column=idx + 1, sticky="w", padx=(0, 6))
            self._iv_axis_controls[f"transform_{mode}"] = chk

        self.graph_iv = SweepGraph(top); self.graph_iv.pack(fill=tk.BOTH, expand=True)
        x_combo.configure(values=self.graph_iv.axis_choices)
        y_combo.configure(values=self.graph_iv.axis_choices)
        x_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_graph())
        y_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_graph())
        self._iv_axis_controls.update({
            "x_combo": x_combo,
            "y_combo": y_combo,
            "logx": logx_chk,
            "logy": logy_chk,
        })
        self._update_transform_controls()

        # Row 1: three compact columns
        src = ttk.Frame(parent, style="White.TFrame"); src.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=4)
        sweep = ttk.Frame(parent, style="White.TFrame"); sweep.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        cycle = ttk.Frame(parent, style="White.TFrame"); cycle.grid(row=1, column=2, sticky="nsew", padx=(4, 8), pady=4)

        self._build_source_settings(src)
        self._build_sweep_settings(sweep)
        self._build_cycle_settings(cycle)

        # Row 2: Result table across all columns
        self.result_tree_iv = self._build_result_table(parent, row=2, col=0, colspan=3)

    def _toggle_transform_mode(self, mode: str) -> None:
        flag = self.transform_flags_iv.get(mode)
        if flag is None:
            return
        enabled = bool(flag.get())
        if enabled:
            for key, other in self.transform_flags_iv.items():
                if key != mode:
                    other.set(False)
            self.transform_mode_var_iv.set(mode)
        else:
            self.transform_mode_var_iv.set("none")
        self._update_transform_controls()
        self._refresh_graph()
        self._save_config()

    def _update_transform_controls(self) -> None:
        mode = self.transform_mode_var_iv.get()
        x_combo = self._iv_axis_controls.get("x_combo")
        y_combo = self._iv_axis_controls.get("y_combo")
        logx = self._iv_axis_controls.get("logx")
        logy = self._iv_axis_controls.get("logy")

        if isinstance(x_combo, ttk.Combobox):
            if mode == "none":
                x_combo.state(["!disabled"])
                x_combo.configure(state="readonly")
            else:
                x_combo.state(["disabled"])
        if isinstance(y_combo, ttk.Combobox):
            if mode == "none":
                y_combo.state(["!disabled"])
                y_combo.configure(state="readonly")
            else:
                y_combo.state(["disabled"])
        if isinstance(logx, ttk.Checkbutton):
            if mode == "none":
                logx.state(["!disabled"])
            else:
                logx.state(["disabled"])
        if isinstance(logy, ttk.Checkbutton):
            if mode == "none":
                logy.state(["!disabled"])
            else:
                logy.state(["disabled"])

        if mode != "none":
            for key, flag in self.transform_flags_iv.items():
                flag.set(key == mode)
            self.logx_var_iv.set(False)
            self.logy_var_iv.set(False)
        if self.graph_iv is not None:
            self.graph_iv.set_transform(mode if mode != "none" else "none")

    def _auto_mode_labels(self) -> Dict[str, str]:
        return {
            "manual": "手动设置",
            "from_step": "按步长算点数",
            "from_points": "按点数算步进",
        }

    def _on_auto_mode_changed(self) -> None:
        label = self._auto_mode_display_var.get()
        reverse = {v: k for k, v in self._auto_mode_labels().items()}
        code = reverse.get(label, "manual")
        self.auto_points_mode_var.set(code)
        self._update_auto_entry_states()
        self._refresh_auto_calculation()
        self._save_config()

    def _update_auto_entry_states(self) -> None:
        mode = self.auto_points_mode_var.get()
        if self._step_entry is not None:
            if mode == "from_points":
                self._step_entry.state(["disabled"])
            else:
                self._step_entry.state(["!disabled"])
        if self._points_entry is not None:
            if mode == "from_step":
                self._points_entry.state(["disabled"])
            else:
                self._points_entry.state(["!disabled"])

    def _refresh_auto_calculation(self) -> None:
        if self._auto_calc_label is None:
            return
        mode_labels = self._auto_mode_labels()
        self._auto_mode_display_var.set(mode_labels.get(self.auto_points_mode_var.get(), mode_labels["manual"]))
        self._update_auto_entry_states()
        try:
            s = self.instrument.settings
            start_from = self.sweep_start_var.get()
            if start_from == "从零点":
                origin, dest = 0.0, float(s.stop_level)
            elif start_from == "从终点":
                origin, dest = float(s.stop_level), float(s.start_level)
            else:
                origin, dest = float(s.start_level), float(s.stop_level)
            span = dest - origin
            mode = self.auto_points_mode_var.get()
            if mode == "from_step":
                step = float(s.step)
                if math.isclose(step, 0.0, rel_tol=1e-9, abs_tol=1e-12):
                    self._auto_calc_label.config(text="自动点数：步进为 0")
                    return
                n_base = int(abs(span / step)) + 1 if not math.isclose(step, 0.0, abs_tol=1e-15) else 1
                n_display = n_base
                if self.sweep_traj_var.get().startswith("往返") and n_base > 1:
                    n_display = n_base * 2 - 1
                self._auto_calc_label.config(text=f"自动点数：{n_display}")
                if self._points_entry is not None:
                    was_disabled = self._points_entry.instate(["disabled"])
                    if was_disabled:
                        self._points_entry.state(["!disabled"])
                    self._points_entry.delete(0, tk.END)
                    self._points_entry.insert(0, str(n_base))
                    if was_disabled:
                        self._points_entry.state(["disabled"])
                s.sweep_points = n_base
            elif mode == "from_points":
                points = max(int(s.sweep_points), 1)
                if points <= 1 or math.isclose(span, 0.0, abs_tol=1e-15):
                    step = 0.0
                    self._auto_calc_label.config(text="自动步进：范围为 0")
                else:
                    step = span / (points - 1)
                    direction = 1 if dest >= origin else -1
                    if direction > 0 and step < 0: step = abs(step)
                    if direction < 0 and step > 0: step = -abs(step)
                    self._auto_calc_label.config(text=f"自动步进：{fmt_e(step, 3)}")
                if self._step_entry is not None:
                    was_disabled = self._step_entry.instate(["disabled"])
                    if was_disabled:
                        self._step_entry.state(["!disabled"])
                    self._step_entry.delete(0, tk.END)
                    self._step_entry.insert(0, f"{step}")
                    if was_disabled:
                        self._step_entry.state(["disabled"])
                s.step = float(step)
            else:
                self._auto_calc_label.config(text="自动计算：--")
                return
        except Exception:
            self._auto_calc_label.config(text="自动计算：--")
            return
        self._save_config()

    # ----------------------- Right: Vt/It tabs (grid) -------------------
    def _build_time_tab(self, parent: ttk.Frame, *, mode: str) -> None:
        parent.rowconfigure(0, weight=6)
        parent.rowconfigure(1, weight=2)
        parent.rowconfigure(2, weight=3)
        for c in (0, 1, 2):
            parent.columnconfigure(c, weight=1)

        # Row 0 graph block
        top = ttk.Frame(parent, style="White.TFrame")
        top.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=8, pady=(6, 4))

        if mode == "Vt":
            x_var, y_var = self.x_axis_var_vt, self.y_axis_var_vt
            y_default = "电压 (V)"; logx, logy = self.logx_var_vt, self.logy_var_vt
        else:
            x_var, y_var = self.x_axis_var_it, self.y_axis_var_it
            y_default = "电流 (A)"; logx, logy = self.logx_var_it, self.logy_var_it
        x_var.set("时间 (s)"); y_var.set(y_default)

        axis = ttk.Frame(top, style="White.TFrame")
        axis.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(axis, text="横轴").pack(side=tk.LEFT)
        ttk.Combobox(axis, textvariable=x_var, state="readonly", width=14, values=list(SweepGraph.AXIS_FIELDS.keys())).pack(side=tk.LEFT, padx=(6, 18))
        ttk.Label(axis, text="纵轴").pack(side=tk.LEFT)
        ttk.Combobox(axis, textvariable=y_var, state="readonly", width=14, values=list(SweepGraph.AXIS_FIELDS.keys())).pack(side=tk.LEFT, padx=(6, 18))
        ttk.Checkbutton(axis, text="X 取 log10", variable=logx, command=self._refresh_graph).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(axis, text="Y 取 log10", variable=logy, command=self._refresh_graph).pack(side=tk.LEFT)

        graph = SweepGraph(top); graph.pack(fill=tk.BOTH, expand=True)

        # Row 1 params compact (left two columns)
        p = ttk.LabelFrame(parent, text="时间采样参数", style="White.TFrame")
        p.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=(8, 4), pady=(0, 4))
        for c in (1, 3, 5): p.columnconfigure(c, weight=1)

        if mode == "Vt":
            self.vt_level_var = getattr(self, "vt_level_var", tk.DoubleVar(value=1.0))
            self.vt_dur_var = getattr(self, "vt_dur_var", tk.DoubleVar(value=5.0))
            self.vt_intv_var = getattr(self, "vt_intv_var", tk.DoubleVar(value=0.5))

            ttk.Label(p, text="固定电平").grid(row=0, column=0, sticky="w", padx=6, pady=3)
            ttk.Entry(p, textvariable=self.vt_level_var).grid(row=0, column=1, sticky="ew", padx=6, pady=3)
            ttk.Label(p, text="总时长(s)").grid(row=0, column=2, sticky="e", padx=6, pady=3)
            ttk.Entry(p, textvariable=self.vt_dur_var).grid(row=0, column=3, sticky="ew", padx=6, pady=3)
            ttk.Label(p, text="采样间隔(s)").grid(row=0, column=4, sticky="e", padx=6, pady=3)
            ttk.Entry(p, textvariable=self.vt_intv_var).grid(row=0, column=5, sticky="ew", padx=6, pady=3)
        else:
            self.it_level_var = getattr(self, "it_level_var", tk.DoubleVar(value=0.01))
            self.it_dur_var = getattr(self, "it_dur_var", tk.DoubleVar(value=5.0))
            self.it_intv_var = getattr(self, "it_intv_var", tk.DoubleVar(value=0.5))

            ttk.Label(p, text="固定电平").grid(row=0, column=0, sticky="w", padx=6, pady=3)
            ttk.Entry(p, textvariable=self.it_level_var).grid(row=0, column=1, sticky="ew", padx=6, pady=3)
            ttk.Label(p, text="总时长(s)").grid(row=0, column=2, sticky="e", padx=6, pady=3)
            ttk.Entry(p, textvariable=self.it_dur_var).grid(row=0, column=3, sticky="ew", padx=6, pady=3)
            ttk.Label(p, text="采样间隔(s)").grid(row=0, column=4, sticky="e", padx=6, pady=3)
            ttk.Entry(p, textvariable=self.it_intv_var).grid(row=0, column=5, sticky="ew", padx=6, pady=3)

        # Row 1 right: compact cycle
        c1 = ttk.Frame(parent, style="White.TFrame"); c1.grid(row=1, column=2, sticky="nsew", padx=(4, 8), pady=(0, 4))
        self._build_cycle_settings(c1)

        # Row 2: result
        table = self._build_result_table(parent, row=2, col=0, colspan=3)
        if mode == "Vt":
            self.graph_vt = graph; self.result_tree_vt = table
        else:
            self.graph_it = graph; self.result_tree_it = table

    # ------------------------- Params builders --------------------------
    def _build_source_settings(self, parent: ttk.Frame) -> None:
        s = ttk.LabelFrame(parent, text="源表参数", style="White.TFrame")
        s.pack(fill=tk.BOTH, expand=True)
        # row0
        ttk.Label(s, text="通道").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        ch = ttk.Combobox(s, values=["Channel A", "Channel B"], state="readonly", width=10)
        ch.grid(row=0, column=1, sticky="w", padx=6, pady=3)
        ch.current(["Channel A", "Channel B"].index(self.instrument.settings.channel))
        ch.bind("<<ComboboxSelected>>", lambda e: self._update_setting("channel", ch.get()))

        ttk.Label(s, text="工作模式").grid(row=0, column=2, sticky="e", padx=6, pady=3)
        md = ttk.Combobox(s, values=["电流源", "电压源"], state="readonly", width=10)
        md.grid(row=0, column=3, sticky="w", padx=6, pady=3)
        md.current(["电流源", "电压源"].index(self.instrument.settings.mode))
        md.bind("<<ComboboxSelected>>", lambda e: self._update_setting("mode", md.get()))

        ar_var = tk.IntVar(value=1 if self.instrument.settings.autorange else 0)
        ttk.Checkbutton(s, text="自动量程", variable=ar_var, command=lambda: self._update_setting("autorange", bool(ar_var.get()))).grid(row=0, column=4, sticky="w", padx=6, pady=3)

        # row1
        ttk.Label(s, text="NPLC").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        nplc = ttk.Entry(s, width=8); nplc.insert(0, str(self.instrument.settings.nplc))
        nplc.grid(row=1, column=1, sticky="w", padx=6, pady=3)
        nplc.bind("<FocusOut>", lambda e: self._safe_float_update("nplc", nplc.get(), default=1.0))

        ttk.Label(s, text="触发延时(ms)").grid(row=1, column=2, sticky="e", padx=6, pady=3)
        d = ttk.Entry(s, width=10); d.insert(0, str(self.instrument.settings.trigger_delay_ms))
        d.grid(row=1, column=3, sticky="w", padx=6, pady=3)
        d.bind("<FocusOut>", lambda e: self._safe_float_update("trigger_delay_ms", d.get(), default=10.0))

        # row2
        ttk.Label(s, text="合规电压(V)").grid(row=2, column=0, sticky="w", padx=6, pady=3)
        cv = ttk.Entry(s, width=10); cv.insert(0, str(self.instrument.settings.compliance_voltage))
        cv.grid(row=2, column=1, sticky="w", padx=6, pady=3)
        cv.bind("<FocusOut>", lambda e: self._safe_float_update("compliance_voltage", cv.get(), default=10.0))

        ttk.Label(s, text="钳位电流(A)").grid(row=2, column=2, sticky="e", padx=6, pady=3)
        ci = ttk.Entry(s, width=10); ci.insert(0, str(self.instrument.settings.compliance_current))
        ci.grid(row=2, column=3, sticky="w", padx=6, pady=3)
        ci.bind("<FocusOut>", lambda e: self._safe_float_update("compliance_current", ci.get(), default=0.01))

        for c in (1, 3):
            s.columnconfigure(c, weight=1)

    def _build_sweep_settings(self, parent: ttk.Frame) -> None:
        sw = ttk.LabelFrame(parent, text="I/V 扫描参数", style="White.TFrame")
        sw.pack(fill=tk.BOTH, expand=True)

        # row0 trajectory + start
        ttk.Label(sw, text="轨迹").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        traj = ttk.Combobox(sw, state="readonly", width=12, values=["单向", "往返(Up/Down)"], textvariable=self.sweep_traj_var)
        traj.grid(row=0, column=1, sticky="ew", padx=6, pady=3)
        traj.bind("<<ComboboxSelected>>", lambda _e: (self._save_config(), self._refresh_auto_calculation()))

        ttk.Label(sw, text="起始位置").grid(row=0, column=2, sticky="e", padx=6, pady=3)
        start_from = ttk.Combobox(sw, state="readonly", width=12, values=["从起点", "从终点", "从零点"], textvariable=self.sweep_start_var)
        start_from.grid(row=0, column=3, sticky="ew", padx=6, pady=3)
        start_from.bind("<<ComboboxSelected>>", lambda _e: (self._save_config(), self._refresh_auto_calculation()))

        # row1~
        ttk.Label(sw, text="点数计算").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        auto_options = [
            "手动设置",
            "按步长算点数",
            "按点数算步进",
        ]
        auto_combo = ttk.Combobox(sw, textvariable=self._auto_mode_display_var, state="readonly", values=auto_options, width=16)
        auto_combo.grid(row=1, column=1, columnspan=3, sticky="ew", padx=6, pady=3)
        auto_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_auto_mode_changed())

        ttk.Label(sw, text="起始值").grid(row=2, column=0, sticky="w", padx=6, pady=3)
        start_e = ttk.Entry(sw, width=12); start_e.insert(0, str(self.instrument.settings.start_level))
        start_e.grid(row=2, column=1, sticky="ew", padx=6, pady=3)
        start_e.bind("<FocusOut>", lambda e: (self._update_numeric_setting("start_level", start_e.get()), self._refresh_auto_calculation()))

        ttk.Label(sw, text="终止值").grid(row=2, column=2, sticky="e", padx=6, pady=3)
        stop_e = ttk.Entry(sw, width=12); stop_e.insert(0, str(self.instrument.settings.stop_level))
        stop_e.grid(row=2, column=3, sticky="ew", padx=6, pady=3)
        stop_e.bind("<FocusOut>", lambda e: (self._update_numeric_setting("stop_level", stop_e.get()), self._refresh_auto_calculation()))

        ttk.Label(sw, text="步进").grid(row=3, column=0, sticky="w", padx=6, pady=3)
        step_e = ttk.Entry(sw, width=12); step_e.insert(0, str(self.instrument.settings.step))
        step_e.grid(row=3, column=1, sticky="ew", padx=6, pady=3)
        step_e.bind("<FocusOut>", lambda e: (self._update_numeric_setting("step", step_e.get()), self._refresh_auto_calculation()))

        ttk.Label(sw, text="点数").grid(row=3, column=2, sticky="e", padx=6, pady=3)
        points_entry = ttk.Entry(sw, width=12); points_entry.insert(0, str(self.instrument.settings.sweep_points))
        points_entry.grid(row=3, column=3, sticky="ew", padx=6, pady=3)
        points_entry.bind("<FocusOut>", lambda e: (self._update_numeric_setting("sweep_points", points_entry.get()), self._refresh_auto_calculation()))

        points_val = ttk.Label(sw, text="自动点数：--", foreground="#777")
        points_val.grid(row=4, column=0, columnspan=4, sticky="w", padx=6, pady=(0, 3))

        for c in (1, 3):
            sw.columnconfigure(c, weight=1)
        self._points_entry = points_entry
        self._step_entry = step_e
        self._auto_calc_label = points_val
        self._refresh_auto_calculation()

    def _build_cycle_settings(self, parent: ttk.Frame) -> None:
        cyc = ttk.LabelFrame(parent, text="循环控制", style="White.TFrame")
        cyc.pack(fill=tk.BOTH, expand=True)

        ttk.Label(cyc, text="次数").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(cyc, textvariable=self.cycle_count_var, width=8).grid(row=0, column=1, sticky="ew", padx=6, pady=3)

        ttk.Label(cyc, text="间隔(s)").grid(row=0, column=2, sticky="e", padx=6, pady=3)
        ttk.Entry(cyc, textvariable=self.cycle_interval_var, width=8).grid(row=0, column=3, sticky="ew", padx=6, pady=3)

        ttk.Checkbutton(cyc, text="启用移动平均", variable=self.filter_enable_var).grid(row=1, column=0, sticky="w", padx=6, pady=3)
        ttk.Label(cyc, text="窗口N").grid(row=1, column=2, sticky="e", padx=6, pady=3)
        ttk.Entry(cyc, textvariable=self.filter_window_var, width=8).grid(row=1, column=3, sticky="ew", padx=6, pady=3)

        cyc.columnconfigure(1, weight=1); cyc.columnconfigure(3, weight=1)

    def _build_result_table(self, parent: ttk.Frame, *, row: int, col: int, colspan: int = 1) -> ttk.Treeview:
        frame = ttk.LabelFrame(parent, text="测量结果", style="White.TFrame")
        frame.grid(row=row, column=col, columnspan=colspan, sticky="nsew", padx=8, pady=(2, 8))
        parent.rowconfigure(row, weight=3)

        cols = ("序号", "时间", "电压(V)", "电流(A)", "电阻(Ω)", "功率(W)")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=8)
        for c in cols:
            tree.heading(c, text=c)
            width = 80 if c in ("序号", "时间") else 110
            tree.column(c, anchor="center", width=width, stretch=True)
        tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview).pack(side=tk.RIGHT, fill=tk.Y)
        return tree

    # ------------------------ resource & connect ------------------------
    def _update_resource_options(self, resources: List[VisaResourceInfo]) -> None:
        values = ["内置模拟器"]; self._resource_map.clear()
        for info in resources:
            label = info.display_name() if (info.model or info.idn) else info.resource
            cand = label; sfx = 1
            while cand in values: sfx += 1; cand = f"{label} #{sfx}"
            values.append(cand); self._resource_map[cand] = info
        self.resource_combo.configure(values=values)
        if self.resource_var.get() not in values: self.resource_var.set(values[0])
        self._on_resource_selected()

    def _on_resource_selected(self) -> None:
        sel = self.resource_var.get()
        if sel in self._resource_map:
            info = self._resource_map[sel]
            self.resource_info_var.set(f"{(info.manufacturer or '未知厂商')} {(info.model or '未知型号')}\n资源: {info.resource}\n序列号: {info.serial or '未提供'}")
        else:
            self.resource_info_var.set("使用内置模拟器")

    def scan_instruments(self) -> None:
        # Fast scan with concurrent *IDN?* queries
        if pyvisa is None:
            messagebox.showwarning("未检测到PyVISA", "需要安装 PyVISA + NI-VISA 才能连真机。"); return
        self.summary_var.set("正在快速扫描..."); self.update_idletasks()
        res = discover_visa_resources_fast(timeout_s=0.12, with_idn=True, max_workers=12)
        self._update_resource_options(res)
        if res: self.summary_var.set(f"发现 {len(res)} 台设备"); self._log("扫描到设备: " + ", ".join((x.model or x.resource) for x in res))
        else: self.summary_var.set("未发现设备，继续使用模拟器"); self._log("未发现任何 VISA 设备")

    def connect_instrument(self) -> None:
        if self.instrument.is_connected:
            messagebox.showinfo("提示", "仪器已连接"); return
        sel = self.resource_var.get()
        inst = VisaInstrument(self._resource_map[sel]) if sel in self._resource_map else InstrumentSimulator()
        inst.settings = self.instrument.settings
        try: inst.connect()
        except Exception as exc:
            messagebox.showerror("错误", f"连接失败: {exc}"); return
        self.instrument = inst  # type: ignore
        status = inst.info.display_name() if isinstance(inst, VisaInstrument) else "模拟器"
        self.device_status_var.set(f"已连接 ({status})"); self.summary_var.set("仪器连接成功"); self._log(f"已连接：{status}")

    def disconnect_instrument(self) -> None:
        if not self.instrument.is_connected:
            messagebox.showinfo("提示", "仪器未连接"); return
        try: self.instrument.disconnect()
        except Exception as exc:
            messagebox.showerror("错误", f"断开失败: {exc}"); self._log(f"断开失败: {exc}"); return
        self.device_status_var.set("未连接"); self.summary_var.set("仪器已断开"); self._log("仪器连接断开")

    # ----------------------------- Run logic ----------------------------
    def _on_tab_changed(self) -> None:
        tab = self.notebook.tab(self.notebook.select(), "text")
        if tab == "I/V测量":
            self.graph_active = self.graph_iv; self.result_tree_active = self.result_tree_iv; self._active_mode = "IV"
        elif tab == "V/t测试":
            self.graph_active = self.graph_vt; self.result_tree_active = self.result_tree_vt; self._active_mode = "Vt"
        else:
            self.graph_active = self.graph_it; self.result_tree_active = self.result_tree_it; self._active_mode = "It"
        self._refresh_graph()

    def start_measurement(self) -> None:
        if not self.instrument.is_connected:
            messagebox.showwarning("警告", "请先连接仪器"); return
        if self._thread and self._thread.is_alive():
            messagebox.showinfo("提示", "测量正在进行中"); return

        self.summary_var.set("测量进行中..."); self._stop.clear(); self.measurements.clear()
        if self.result_tree_active is not None:
            for it in self.result_tree_active.get_children(): self.result_tree_active.delete(it)
        self._refresh_graph()

        try:
            if self._active_mode == "IV":
                self._setpoints = self._compute_setpoints()
                if not self._setpoints: raise ValueError("无法根据当前参数计算扫描点")
                self.instrument.prepare_measurement(self._setpoints)  # type: ignore
                self._log(f"I/V 扫描：{len(self._setpoints)} 点，范围 {fmt_e(self._setpoints[0])} → {fmt_e(self._setpoints[-1])}，轨迹={self.sweep_traj_var.get()}，起始={self.sweep_start_var.get()}")
                dwell = max(self.instrument.settings.trigger_delay_ms / 1000.0, 0.02)
            else:
                # Vt / It
                if self._active_mode == "Vt":
                    level = float(getattr(self, "vt_level_var").get())
                    duration = max(float(getattr(self, "vt_dur_var").get()), 0.0)
                    interval = max(float(getattr(self, "vt_intv_var").get()), 0.02)
                else:
                    level = float(getattr(self, "it_level_var").get())
                    duration = max(float(getattr(self, "it_dur_var").get()), 0.0)
                    interval = max(float(getattr(self, "it_intv_var").get()), 0.02)
                points = max(int(duration / interval) + 1, 1)
                self._setpoints = [level] * points
                self.instrument.prepare_measurement(self._setpoints)  # type: ignore
                self._log(f"{self._active_mode}：固定 {fmt_e(level)}，时长 {duration:g}s，间隔 {interval:g}s，点数 {points}")
                dwell = interval
        except Exception as exc:
            messagebox.showerror("错误", f"初始化失败: {exc}"); self.summary_var.set("初始化失败"); self._log(f"初始化失败: {exc}")
            self._setpoints = []; return

        # progress init + ETA runtime marker
        total_cycles = max(int(self.cycle_count_var.get()), 1)
        total_points = len(self._setpoints) * total_cycles
        self._total_points_planned = total_points
        self._run_started_at = time.perf_counter()
        self._reset_progress(total_points)
        self._update_eta_dynamic(done_points=0)
        self._current_cycle = 1; self.start_btn.state(["disabled"]); self.stop_btn.state(["!disabled"])
        self._thread = threading.Thread(target=self._run_cycles, daemon=True); self._thread.start()

    def _run_cycles(self) -> None:
        total_cycles = max(int(self.cycle_count_var.get()), 1)
        try:
            while self._current_cycle <= total_cycles and not self._stop.is_set():
                self._run_one_measurement_cycle()
                if self._stop.is_set(): break
                # 自动保存每个循环
                self.after(0, lambda cyc=self._current_cycle: self._auto_save_final(suffix=f"_cycle{cyc}"))
                if self._current_cycle < total_cycles:
                    wait_s = max(float(self.cycle_interval_var.get()), 0.0)
                    steps = int(wait_s * 10)
                    for k in range(steps):
                        if self._stop.is_set(): break
                        # 等待期间 ETA 暂不基于“点”，直接显示倒计时
                        remain = (steps - k - 1) / 10.0
                        self.after(0, lambda r=remain: self.eta_var.set(f"间隔 {r:.1f}s"))
                        time.sleep(0.1)
                self._current_cycle += 1
        finally:
            self.after(0, self._finish_all_cycles)

    def _run_one_measurement_cycle(self) -> None:
        if self._active_mode == "IV":
            dwell = max(self.instrument.settings.trigger_delay_ms / 1000.0, 0.02)
        elif self._active_mode == "Vt":
            dwell = max(float(getattr(self, "vt_intv_var").get()), 0.02)
        else:
            dwell = max(float(getattr(self, "it_intv_var").get()), 0.02)

        total_points = len(self._setpoints)
        try:
            for i, lvl in enumerate(self._setpoints, start=1):
                if self._stop.is_set(): break
                pt = self.instrument.generate_point(len(self.measurements) + 1, lvl)  # type: ignore
                # Optional moving average
                if self.filter_enable_var.get():
                    win = max(int(self.filter_window_var.get()), 1)
                    window = self.measurements[-(win-1):] + [pt] if win > 1 else [pt]
                    v_vals = [p.voltage for p in window]; i_vals = [p.current for p in window]
                    pt = MeasurementPoint(index=pt.index, timestamp=pt.timestamp,
                                          voltage=sum(v_vals)/len(v_vals), current=sum(i_vals)/len(i_vals))
                self.measurements.append(pt)
                self.after(0, self._append_point, pt)

                # progress + ETA (dynamic by rate)
                done_points = (self._current_cycle - 1) * total_points + i
                self.after(0, lambda d=done_points: self._set_progress(d))
                self.after(0, lambda dp=done_points: self._update_eta_dynamic(done_points=dp))

                time.sleep(dwell)
        except Exception as exc:
            self.after(0, lambda e=exc: self._measurement_error(e))

    def _finish_all_cycles(self) -> None:
        self.start_btn.state(["!disabled"]); self.stop_btn.state(["disabled"])
        if self._stop.is_set():
            self.summary_var.set("测量已停止"); self._log("测量被用户中断")
        else:
            self.summary_var.set("测量完成（已自动保存）"); self._log("测量完成"); self._auto_save_final()
        self._stop.clear(); self._setpoints = []; self._set_progress(self.progress.maximum)  # type: ignore

    def stop_measurement(self) -> None:
        if self._thread and self._thread.is_alive():
            self._stop.set()
            try: self.instrument.abort_measurement()  # type: ignore
            except Exception: pass
        self.start_btn.state(["!disabled"]); self.stop_btn.state(["disabled"])
        self.summary_var.set("测量已停止"); self._log("测量停止命令已执行")

    def _append_point(self, pt: MeasurementPoint) -> None:
        tstr = pt.timestamp.strftime("%H:%M:%S")
        res = "∞" if math.isinf(pt.resistance) else fmt_e(pt.resistance, 3)
        row = (pt.index, tstr, fmt_e(pt.voltage, 3), fmt_e(pt.current, 3), res, fmt_e(pt.power, 3))
        if self.result_tree_active is not None:
            self.result_tree_active.insert("", tk.END, values=row)
        self._refresh_graph(); self._update_summary()
        if self.snapshots_var.get(): self._auto_save_snapshot()

    # ETA based on user's formula: remaining = (N_total - N_done) / (N_done / elapsed_time)
    def _update_eta_dynamic(self, *, done_points: int) -> None:
        if self._run_started_at is None or done_points <= 0:
            self.eta_var.set("—"); return
        elapsed = max(time.perf_counter() - self._run_started_at, 1e-9)
        rate = done_points / elapsed  # points per second
        remaining_pts = max(self._total_points_planned - done_points, 0)
        if rate <= 1e-9:
            self.eta_var.set("—")
        else:
            remain_s = remaining_pts / rate
            self.eta_var.set(f"剩余 {remain_s:.1f}s")

    def _active_axis_labels(self) -> Tuple[str, str, bool, bool, str]:
        if self._active_mode == "IV":
            transform = self.transform_mode_var_iv.get()
            if transform != "none":
                return "电压 (V)", "电流 (A)", False, False, transform
            return (
                self.x_axis_var_iv.get(),
                self.y_axis_var_iv.get(),
                self.logx_var_iv.get(),
                self.logy_var_iv.get(),
                "none",
            )
        if self._active_mode == "Vt":
            return (
                self.x_axis_var_vt.get(),
                self.y_axis_var_vt.get(),
                self.logx_var_vt.get(),
                self.logy_var_vt.get(),
                "none",
            )
        return (
            self.x_axis_var_it.get(),
            self.y_axis_var_it.get(),
            self.logx_var_it.get(),
            self.logy_var_it.get(),
            "none",
        )

    def _refresh_graph(self) -> None:
        if self.graph_active is None:
            if self._active_mode == "IV": self.graph_active = self.graph_iv
            elif self._active_mode == "Vt": self.graph_active = self.graph_vt
            else: self.graph_active = self.graph_it
        if self.graph_active is None: return
        xlbl, ylbl, logx, logy, transform = self._active_axis_labels()
        self.graph_active.set_transform(transform)
        self.graph_active.set_log(logx, logy)
        self.graph_active.set_data(self.measurements, x_axis=xlbl, y_axis=ylbl)

    def _update_summary(self) -> None:
        if not self.measurements:
            self.summary_var.set("准备就绪"); return
        v = [p.voltage for p in self.measurements]; i = [p.current for p in self.measurements]
        self.summary_var.set(f"点数 {len(self.measurements)} | ⟨V⟩={fmt_e(statistics.fmean(v),3)} V | ⟨I⟩={fmt_e(statistics.fmean(i),3)} A")

    def _reset_progress(self, total_points: int) -> None:
        if total_points <= 0: total_points = 1
        self.progress.configure(maximum=float(total_points))
        self.progress_var.set(0.0)

    def _set_progress(self, done_points: float) -> None:
        try:
            self.progress_var.set(float(done_points))
        except Exception:
            pass

    # --------------------------- setpoints utils ------------------------
    def _compute_setpoints(self) -> List[float]:
        s = self.instrument.settings
        start_from = self.sweep_start_var.get()  # 从起点 / 从终点 / 从零点
        traj = self.sweep_traj_var.get()         # 单向 / 往返(Up/Down)

        if start_from == "从零点":
            origin = 0.0
            dest = float(s.stop_level)
        elif start_from == "从终点":
            origin = float(s.stop_level); dest = float(s.start_level)
        else:
            origin = float(s.start_level); dest = float(s.stop_level)

        mode = self.auto_points_mode_var.get()
        step = float(s.step)
        points = max(int(s.sweep_points), 1)
        span = dest - origin
        if mode == "from_points":
            if points <= 1 or math.isclose(span, 0.0, abs_tol=1e-15):
                step = 0.0
            else:
                step = span / (points - 1)
        elif mode == "from_step":
            if math.isclose(step, 0.0, rel_tol=1e-9, abs_tol=1e-12):
                if math.isclose(span, 0.0, abs_tol=1e-15):
                    step = 0.0
                else:
                    step = span / 10.0
        else:  # manual
            if math.isclose(step, 0.0, rel_tol=1e-9, abs_tol=1e-12):
                if points <= 1 or math.isclose(span, 0.0, abs_tol=1e-15):
                    step = 0.0
                else:
                    step = span / max(points - 1, 1)

        direction = 1 if dest >= origin else -1
        if direction > 0 and step < 0: step = abs(step)
        if direction < 0 and step > 0: step = -abs(step)

        if mode == "from_step":
            if math.isclose(step, 0.0, abs_tol=1e-15):
                n = 1
            else:
                n = int(abs(span / step)) + 1
        else:
            n = points
        if n <= 0:
            n = 1

        if n <= 1 or math.isclose(dest, origin, abs_tol=1e-15):
            base = [origin]
        else:
            base = [origin + i * step for i in range(n - 1)]
            base.append(dest)

        if traj.startswith("往返"):
            back = list(reversed(base[:-1]))
            base = base + back

        return base

    # ----------------------------- files --------------------------------
    def select_directory(self) -> None:
        d = filedialog.askdirectory()
        if d:
            self.file_path_var.set(d); self._log(f"选择保存目录: {d}"); self._save_config()

    def _auto_save_snapshot(self) -> None:
        d = self.file_path_var.get()
        if not d: return
        p = Path(d); p.mkdir(parents=True, exist_ok=True)
        self._write_csv(p / f"snapshot_{now_stamp()}.csv", silent=True)

    def _auto_save_final(self, *, suffix: str = "") -> None:
        d = self.file_path_var.get()
        if not d:
            self.select_directory(); d = self.file_path_var.get()
            if not d: self._log("自动保存跳过：未选择目录"); return
        p = Path(d); p.mkdir(parents=True, exist_ok=True)
        safe = (self.measure_name_var.get().strip() or "measure").replace(" ", "_")
        filename = p / f"{safe}{suffix}_{now_stamp()}.csv"
        self._write_csv(filename, silent=True)
        self._log(f"结果已保存：{filename}")

    def _write_csv(self, filename: Path, *, silent: bool = False) -> None:
        with filename.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["index", "timestamp", "voltage", "current", "resistance", "power"])
            for pt in self.measurements:
                w.writerow([pt.index, pt.timestamp.isoformat(),
                            f"{pt.voltage:.6f}", f"{pt.current:.6f}",
                            "inf" if math.isinf(pt.resistance) else f"{pt.resistance:.6f}",
                            f"{pt.power:.6f}"])
        if not silent: self._log(f"CSV 已写入: {filename}")

    # ----------------------------- config -------------------------------
    def _save_config(self) -> None:
        try:
            cfg = {
                "instrument_settings": asdict(self.instrument.settings),
                "ui": {
                    "file_path": self.file_path_var.get(),
                    "measure_name": self.measure_name_var.get(),
                    "sample_name": self.sample_name_var.get(),
                    "operator": self.operator_var.get(),
                    "mode": self._active_mode,
                    "auto_points_mode": self.auto_points_mode_var.get(),
                    "filter_enable": self.filter_enable_var.get(),
                    "filter_window": int(self.filter_window_var.get()),
                    "cycle_count": int(self.cycle_count_var.get()),
                    "cycle_interval": float(self.cycle_interval_var.get()),
                    "sweep_traj": self.sweep_traj_var.get(),
                    "sweep_start": self.sweep_start_var.get(),
                    "transform_mode_iv": self.transform_mode_var_iv.get(),
                    "axes": {
                        "iv": [self.x_axis_var_iv.get(), self.y_axis_var_iv.get(), self.logx_var_iv.get(), self.logy_var_iv.get()],
                        "vt": [self.x_axis_var_vt.get(), self.y_axis_var_vt.get(), self.logx_var_vt.get(), self.logy_var_vt.get()],
                        "it": [self.x_axis_var_it.get(), self.y_axis_var_it.get(), self.logx_var_it.get(), self.logy_var_it.get()],
                    },
                },
            }
            with open(CFG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            self._log(f"保存配置失败：{exc}")

    def _load_config_then_autoconnect(self) -> None:
        # load config
        if CFG_PATH.exists():
            try:
                with open(CFG_PATH, "r", encoding="utf-8") as f: cfg = json.load(f)
                s = cfg.get("instrument_settings", {})
                for k, v in s.items():
                    if hasattr(self.instrument.settings, k): setattr(self.instrument.settings, k, v)
                ui = cfg.get("ui", {})
                self.file_path_var.set(ui.get("file_path", ""))
                self.measure_name_var.set(ui.get("measure_name", "默认测试"))
                self.sample_name_var.set(ui.get("sample_name", "Sample-001"))
                self.operator_var.set(ui.get("operator", "Operator"))
                self._active_mode = ui.get("mode", "IV")
                mode = ui.get("auto_points_mode", "from_step")
                if mode not in self._auto_mode_labels():
                    mode = "from_step"
                self.auto_points_mode_var.set(mode)
                self.filter_enable_var.set(ui.get("filter_enable", False))
                self.filter_window_var.set(int(ui.get("filter_window", 5)))
                self.cycle_count_var.set(int(ui.get("cycle_count", 1)))
                self.cycle_interval_var.set(float(ui.get("cycle_interval", 0.0)))
                self.sweep_traj_var.set(ui.get("sweep_traj", "单向"))
                self.sweep_start_var.set(ui.get("sweep_start", "从起点"))
                transform_mode = ui.get("transform_mode_iv", "none")
                if transform_mode not in self.transform_flags_iv and transform_mode != "none":
                    transform_mode = "none"
                self.transform_mode_var_iv.set(transform_mode)
                for key, flag in self.transform_flags_iv.items():
                    flag.set(key == transform_mode)
                axes = ui.get("axes", {})
                if "iv" in axes:
                    self.x_axis_var_iv.set(axes["iv"][0]); self.y_axis_var_iv.set(axes["iv"][1])
                    self.logx_var_iv.set(bool(axes["iv"][2])); self.logy_var_iv.set(bool(axes["iv"][3]))
                if "vt" in axes:
                    self.x_axis_var_vt.set(axes["vt"][0]); self.y_axis_var_vt.set(axes["vt"][1])
                    self.logx_var_vt.set(bool(axes["vt"][2])); self.logy_var_vt.set(bool(axes["vt"][3]))
                if "it" in axes:
                    self.x_axis_var_it.set(axes["it"][0]); self.y_axis_var_it.set(axes["it"][1])
                    self.logx_var_it.set(bool(axes["it"][2])); self.logy_var_it.set(bool(axes["it"][3]))
                self._log("已加载上次配置")
            except Exception as exc:
                self._log(f"加载配置失败：{exc}")
        self._update_transform_controls()
        self._refresh_auto_calculation()
        self._refresh_graph()

        # update resource list quickly (no IDN for UI), then try 2636B autoconnect
        if pyvisa is not None:
            quick = discover_visa_resources_fast(with_idn=False)
            self._update_resource_options(quick)
            # only connect when a 2636B is present
            info = discover_first_2636b(timeout_s=0.15, max_workers=12)
            if info is not None:
                # Put it into combo map so label renders nicely
                label = info.display_name() if (info.model or info.idn) else info.resource
                self._resource_map[label] = info
                vals = list(self.resource_combo.cget("values"));
                if label not in vals: vals.append(label)
                self.resource_combo.configure(values=vals)
                self.resource_var.set(label)
                try:
                    inst = VisaInstrument(info)
                    inst.settings = self.instrument.settings
                    inst.connect()
                    self.instrument = inst  # type: ignore
                    self.device_status_var.set(f"已连接 ({info.display_name()})")
                    self.summary_var.set("已自动连接 2636B")
                    self._log(f"自动连接：{info.display_name()}")
                except Exception as exc:
                    self._log(f"自动连接失败：{exc}")
            else:
                self._log("未发现 2636B；保持未连接/模拟器")
        else:
            self._update_resource_options([])

    # ------------------------------- misc --------------------------------
    def _update_setting(self, field_name: str, value) -> None:
        setattr(self.instrument.settings, field_name, value); self._log(f"更新参数 {field_name} -> {value}"); self._save_config()

    def _safe_float_update(self, field_name: str, value: str, *, default: float) -> None:
        try: numeric = float(value)
        except ValueError:
            messagebox.showerror("无效输入", f"请输入有效数字: {value!r}"); numeric = default
        setattr(self.instrument.settings, field_name, numeric); self._log(f"更新参数 {field_name} -> {numeric}"); self._save_config()

    def _update_numeric_setting(self, field_name: str, value: str) -> None:
        try: numeric = float(value) if field_name != "sweep_points" else int(value)
        except ValueError:
            messagebox.showerror("无效输入", f"请输入有效数字: {value!r}"); return
        setattr(self.instrument.settings, field_name, numeric); self._log(f"更新参数 {field_name} -> {numeric}"); self._save_config()

    def _log(self, msg: str) -> None:
        ts = _dt.datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state=tk.NORMAL); self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_text.see(tk.END); self.log_text.configure(state=tk.DISABLED)

    def on_exit(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive(): self._thread.join(timeout=1.0)
        try: self.instrument.disconnect()
        except Exception: pass
        self._save_config(); self.destroy()


def main() -> None:
    app = MeasurementApp(); app.mainloop()

if __name__ == "__main__":
    main()
