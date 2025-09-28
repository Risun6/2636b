"""2636B Measurement Software replica.

This module implements a desktop application that emulates the user interface shown in
provided screenshots for a 2636B control program.  The implementation is intentionally
self-contained and relies solely on the Python standard library so it can run without
additional dependencies.  The graphical interface is built with :mod:`tkinter` and
includes a simulated data acquisition engine that generates synthetic measurements.

Running the module directly launches the GUI application:

    python main.py

"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import random
import statistics
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pyvisa
    from pyvisa import VisaIOError
except Exception:  # pragma: no cover - pyvisa may be unavailable on CI
    pyvisa = None  # type: ignore[assignment]
    VisaIOError = Exception  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Instrument abstractions
# ---------------------------------------------------------------------------


class InstrumentError(RuntimeError):
    """Base class for instrument related failures."""


@dataclass
class VisaResourceInfo:
    """Metadata discovered during VISA resource scanning."""

    resource: str
    idn: str = ""
    alias: Optional[str] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    serial: Optional[str] = None
    firmware: Optional[str] = None

    def display_name(self) -> str:
        if self.model:
            base = self.model
        elif self.alias:
            base = self.alias
        else:
            base = self.resource
        details = []
        if self.serial:
            details.append(f"SN {self.serial}")
        if self.idn:
            details.append(self.idn)
        info = " | ".join(details)
        return f"{base} ({self.resource})" if not info else f"{base} ({info})"


# ---------------------------------------------------------------------------
# VISA discovery helper
# ---------------------------------------------------------------------------


def discover_visa_resources(timeout: float = 0.5) -> List[VisaResourceInfo]:
    """Scan connected VISA resources and collect identity information."""

    if pyvisa is None:
        return []

    resources: List[VisaResourceInfo] = []
    try:
        rm = pyvisa.ResourceManager()  # type: ignore[call-arg]
    except Exception:  # pragma: no cover - dependent on VISA installation
        return resources

    try:
        for resource_name in rm.list_resources():
            info = VisaResourceInfo(resource=resource_name)
            session = None
            try:
                session = rm.open_resource(resource_name)
                session.timeout = int(timeout * 1000)
                try:
                    alias = rm.resource_info(resource_name).alias
                    info.alias = alias or None
                except Exception:
                    info.alias = None
                try:
                    idn = session.query("*IDN?").strip()
                except Exception:
                    idn = ""
                info.idn = idn
                if idn:
                    parts = [part.strip() for part in idn.split(",")]
                    if parts:
                        info.manufacturer = parts[0]
                    if len(parts) > 1:
                        info.model = parts[1]
                    if len(parts) > 2:
                        info.serial = parts[2]
                    if len(parts) > 3:
                        info.firmware = parts[3]
            except Exception:
                pass
            finally:
                if session is not None:
                    try:
                        session.close()
                    except Exception:
                        pass
            resources.append(info)
    finally:
        try:
            rm.close()
        except Exception:
            pass

    return resources


# ---------------------------------------------------------------------------
# Simulation model
# ---------------------------------------------------------------------------

@dataclass
class MeasurementPoint:
    """Container representing a single simulated measurement point."""

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
class TimeTestContext:
    """Runtime context for time-based measurements (V/t, I/t)."""

    name: str
    required_mode: str
    y_axis_label: str
    target_field: str
    graph: "SweepGraph"
    tree: "ttk.Treeview"
    start_button: "ttk.Button"
    stop_button: "ttk.Button"
    duration_var: "tk.DoubleVar"
    interval_var: "tk.DoubleVar"
    level_var: "tk.DoubleVar"
    x_axis_var: "tk.StringVar"
    y_axis_var: "tk.StringVar"
    status_var: "tk.StringVar"
    metrics_var: "tk.StringVar"
    points: List[MeasurementPoint] = field(default_factory=list)
    thread: Optional[threading.Thread] = None
    stop_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class InstrumentSettings:
    """User configurable settings for the virtual instrument."""

    channel: str = "Channel A"
    mode: str = "电流源"
    autorange: bool = True
    nplc: float = 1.0
    trigger_delay_ms: float = 10.0
    compliance_voltage: float = 10.0
    start_level: float = 0.0
    stop_level: float = 1.0
    step: float = 0.1
    sweep_points: int = 11
    output_enable: bool = True


class InstrumentSimulator:
    """Simulated back-end for the 2636B controller.

    The simulator produces pseudo-random measurements that follow a deterministic pattern
    based on the configured sweep parameters.  The behaviour loosely mimics a voltage or
    current sweep while remaining completely deterministic for a fixed random seed.
    """

    def __init__(self) -> None:
        self.settings = InstrumentSettings()
        self._is_connected = False
        self._start_time: Optional[_dt.datetime] = None
        self._seed = random.Random(2636)
        self._last_level: float = 0.0

    # ------------------------------------------------------------------
    # Instrument management
    # ------------------------------------------------------------------
    def connect(self) -> None:
        time.sleep(0.1)
        self._is_connected = True
        self._start_time = None

    def disconnect(self) -> None:
        self._is_connected = False
        self._start_time = None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    # ------------------------------------------------------------------
    # Measurement routine
    # ------------------------------------------------------------------
    def prepare_measurement(self, setpoints: Sequence[float]) -> None:
        self._last_level = setpoints[0] if setpoints else 0.0
        self.start_measurement()

    def start_measurement(self) -> None:
        if not self._is_connected:
            raise RuntimeError("Instrument not connected")
        self._start_time = _dt.datetime.now()

    def generate_point(self, index: int, level: Optional[float] = None) -> MeasurementPoint:
        if self._start_time is None:
            self._start_time = _dt.datetime.now()

        if level is None:
            level = self.settings.start_level + (index - 1) * self.settings.step
        self._last_level = level

        elapsed = abs(level - self.settings.start_level)
        angle = (elapsed / max(abs(self.settings.stop_level - self.settings.start_level), 0.1)) * math.tau

        if self.settings.mode == "电流源":
            base_current = level
            base_voltage = math.sin(angle) * 5 + base_current * 200
        else:
            base_voltage = level
            base_current = math.cos(angle) * 0.01 + base_voltage / 200

        noise_v = self._seed.uniform(-0.05, 0.05)
        noise_i = self._seed.uniform(-0.0002, 0.0002)

        voltage = base_voltage + noise_v
        current = base_current + noise_i

        timestamp = self._start_time + _dt.timedelta(seconds=max(elapsed * 0.1, 0.1))
        return MeasurementPoint(index=index, timestamp=timestamp, voltage=voltage, current=current)

    def finalize_measurement(self) -> None:
        self._last_level = 0.0
        self._start_time = None

    def abort_measurement(self) -> None:
        self.finalize_measurement()


class VisaInstrument:
    """PyVISA based controller for a physical 2636B instrument."""

    def __init__(self, info: VisaResourceInfo, *, timeout: float = 10.0) -> None:
        self.settings = InstrumentSettings()
        self.info = info
        self.timeout = timeout
        self._rm: Optional["pyvisa.ResourceManager"] = None
        self._resource: Optional["pyvisa.resources.Resource"] = None
        self._is_connected = False
        self._channel = "smua"

    # ------------------------------------------------------------------
    def connect(self) -> None:
        if pyvisa is None:
            raise InstrumentError("未安装 PyVISA，无法连接真实仪器")
        try:
            self._rm = pyvisa.ResourceManager()  # type: ignore[call-arg]
            self._resource = self._rm.open_resource(self.info.resource)
            self._resource.timeout = int(self.timeout * 1000)
            self._resource.write("*CLS")
            self._resource.write("format.data = format.ASCII")
            self._resource.write("format.asciiprecision = 6")
            self._resource.write("format.asciiexponent = 3")
            self._is_connected = True
        except Exception as exc:  # pragma: no cover - hardware specific
            raise InstrumentError(f"连接仪器失败: {exc}") from exc

    def disconnect(self) -> None:
        try:
            self.abort_measurement()
        finally:
            if self._resource is not None:
                try:
                    self._resource.close()
                except Exception:
                    pass
            if self._rm is not None:
                try:
                    self._rm.close()
                except Exception:
                    pass
            self._resource = None
            self._rm = None
            self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    # ------------------------------------------------------------------
    def _write(self, command: str) -> None:
        if self._resource is None:
            raise InstrumentError("仪器未连接")
        try:
            self._resource.write(command)
        except VisaIOError as exc:  # pragma: no cover - hardware specific
            raise InstrumentError(f"仪器通信失败: {exc}") from exc

    def _query(self, command: str) -> str:
        if self._resource is None:
            raise InstrumentError("仪器未连接")
        try:
            return str(self._resource.query(command)).strip()
        except VisaIOError as exc:  # pragma: no cover - hardware specific
            raise InstrumentError(f"仪器通信失败: {exc}") from exc

    def _channel_name(self) -> str:
        return "smua" if self.settings.channel == "Channel A" else "smub"

    def prepare_measurement(self, setpoints: Sequence[float]) -> None:
        if not self._is_connected:
            self.connect()
        self._channel = self._channel_name()
        self._write("errorqueue.clear()")
        self._write("display.clear()")
        self._write(f"{self._channel}.reset()")
        self._write(f"{self._channel}.measure.nplc = {max(self.settings.nplc, 0.01)}")
        self._write(f"{self._channel}.source.output = {self._channel}.OUTPUT_OFF")
        if self.settings.mode == "电流源":
            self._write(f"{self._channel}.source.func = {self._channel}.OUTPUT_DCAMPS")
            if self.settings.autorange:
                self._write(f"{self._channel}.source.autorangei = {self._channel}.AUTORANGE_ON")
            else:
                self._write(f"{self._channel}.source.autorangei = {self._channel}.AUTORANGE_OFF")
            self._write(f"{self._channel}.source.limitv = {abs(self.settings.compliance_voltage)}")
        else:
            self._write(f"{self._channel}.source.func = {self._channel}.OUTPUT_DCVOLTS")
            if self.settings.autorange:
                self._write(f"{self._channel}.source.autorangev = {self._channel}.AUTORANGE_ON")
            else:
                self._write(f"{self._channel}.source.autorangev = {self._channel}.AUTORANGE_OFF")
            self._write(f"{self._channel}.source.limiti = {abs(self.settings.compliance_voltage) / 1000.0}")

        delay = max(self.settings.trigger_delay_ms / 1000.0, 0.0)
        self._write(f"{self._channel}.source.delay = {delay}")
        initial = setpoints[0] if setpoints else 0.0
        if self.settings.mode == "电流源":
            self._write(f"{self._channel}.source.leveli = {initial}")
        else:
            self._write(f"{self._channel}.source.levelv = {initial}")
        self._write(f"{self._channel}.source.output = {self._channel}.OUTPUT_ON")

    def generate_point(self, index: int, level: Optional[float] = None) -> MeasurementPoint:
        if level is None:
            level = self.settings.start_level + (index - 1) * self.settings.step
        if self.settings.mode == "电流源":
            self._write(f"{self._channel}.source.leveli = {level}")
        else:
            self._write(f"{self._channel}.source.levelv = {level}")
        query = (
            "print(string.format('%e,%e', "
            f"{self._channel}.measure.v(), {self._channel}.measure.i()))"
        )
        response = self._query(query)
        try:
            voltage_str, current_str = response.split(",")
            voltage = float(voltage_str)
            current = float(current_str)
        except Exception as exc:  # pragma: no cover - parse failure unlikely
            raise InstrumentError(f"无法解析仪器返回数据: {response}") from exc
        timestamp = _dt.datetime.now()
        return MeasurementPoint(index=index, timestamp=timestamp, voltage=voltage, current=current)

    def finalize_measurement(self) -> None:
        try:
            self._write(f"{self._channel}.source.output = {self._channel}.OUTPUT_OFF")
        except InstrumentError:
            pass

    def abort_measurement(self) -> None:
        try:
            self._write(f"{self._channel}.source.output = {self._channel}.OUTPUT_OFF")
        except InstrumentError:
            pass


# ---------------------------------------------------------------------------
# Graph rendering helper
# ---------------------------------------------------------------------------

class SweepGraph(ttk.Frame):
    """Canvas based plot with configurable axes and robust scaling."""

    AXIS_FIELDS: Dict[str, Tuple[str, str]] = {
        "序号": ("index", ""),
        "时间 (s)": ("time", "s"),
        "电压 (V)": ("voltage", "V"),
        "电流 (A)": ("current", "A"),
        "电阻 (Ω)": ("resistance", "Ω"),
        "功率 (W)": ("power", "W"),
    }

    def __init__(self, master: tk.Widget, *, width: int = 840, height: int = 320) -> None:
        super().__init__(master)
        self.width = width
        self.height = height
        self.canvas = tk.Canvas(self, width=width, height=height, background="#ffffff", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.margin = 50
        self.axis_choices = list(self.AXIS_FIELDS.keys())
        self._x_axis_label = "电压 (V)"
        self._y_axis_label = "电流 (A)"
        self._data: List[Tuple[float, float]] = []
        self._raw_points: List[MeasurementPoint] = []
        self._x_range = (0.0, 1.0)
        self._y_range = (-1.0, 1.0)
        self._time_reference: Optional[_dt.datetime] = None
        self._theme = "light"
        self._colors = self._palette_for_theme(self._theme)
        self.canvas.configure(background=self._colors["background"])
        self.canvas.bind("<Configure>", lambda event: self.redraw())
        self.redraw()

    # ------------------------------------------------------------------
    def set_data(
        self,
        points: Iterable[MeasurementPoint],
        *,
        x_axis: Optional[str] = None,
        y_axis: Optional[str] = None,
    ) -> None:
        if x_axis is not None:
            self._x_axis_label = x_axis
        if y_axis is not None:
            self._y_axis_label = y_axis

        self._raw_points = list(points)
        self._time_reference = None
        self._data = []

        x_field = self.AXIS_FIELDS.get(self._x_axis_label, ("index", ""))[0]
        y_field = self.AXIS_FIELDS.get(self._y_axis_label, ("current", "A"))[0]
        for point in self._raw_points:
            x_value = self._value_from_point(point, x_field)
            y_value = self._value_from_point(point, y_field)
            if not (self._is_valid_number(x_value) and self._is_valid_number(y_value)):
                continue
            self._data.append((x_value, y_value))

        if self._data:
            xs, ys = zip(*self._data)
            self._x_range = self._expand_range(min(xs), max(xs))
            self._y_range = self._expand_range(min(ys), max(ys))
        else:
            self._x_range = (0.0, 1.0)
            self._y_range = (-1.0, 1.0)
        self.redraw()

    # ------------------------------------------------------------------
    def _value_from_point(self, point: MeasurementPoint, field: str) -> float:
        if field == "index":
            return float(point.index)
        if field == "time":
            if self._time_reference is None:
                self._time_reference = point.timestamp
            return (point.timestamp - self._time_reference).total_seconds()
        value = getattr(point, field, None)
        if value is None:
            return float("nan")
        return float(value)

    def _is_valid_number(self, value: float) -> bool:
        return isinstance(value, (int, float)) and math.isfinite(float(value))

    def _expand_range(self, vmin: float, vmax: float) -> Tuple[float, float]:
        if not math.isfinite(vmin) or not math.isfinite(vmax):
            return (-1.0, 1.0)
        if math.isclose(vmin, vmax, rel_tol=1e-9, abs_tol=1e-12):
            pad = abs(vmin) * 0.05 or 1.0
            return (vmin - pad, vmax + pad)
        span = vmax - vmin
        pad = span * 0.1 or 1.0
        return (vmin - pad, vmax + pad)

    def _nice_number(self, value: float) -> float:
        if value <= 0:
            return 1.0
        exponent = math.floor(math.log10(value))
        fraction = value / (10 ** exponent)
        if fraction < 1.5:
            nice_fraction = 1.0
        elif fraction < 3.0:
            nice_fraction = 2.0
        elif fraction < 7.0:
            nice_fraction = 5.0
        else:
            nice_fraction = 10.0
        return nice_fraction * (10 ** exponent)

    def _generate_ticks(self, vmin: float, vmax: float, *, count: int = 6) -> List[Tuple[float, str]]:
        if not math.isfinite(vmin) or not math.isfinite(vmax):
            return []
        if math.isclose(vmax, vmin, rel_tol=1e-9, abs_tol=1e-12):
            vmax = vmin + 1.0
        span = vmax - vmin
        step = self._nice_number(span / max(count - 1, 1))
        start = math.floor(vmin / step) * step
        ticks: List[Tuple[float, str]] = []
        value = start
        while value <= vmax + step * 0.5:
            ticks.append((value, f"{value:.3g}"))
            value += step
        return ticks

    # ------------------------------------------------------------------
    def redraw(self) -> None:
        self.canvas.delete("all")
        width = max(10, self.canvas.winfo_width())
        height = max(10, self.canvas.winfo_height())
        margin = self.margin
        plot_left = margin
        plot_top = margin
        plot_right = width - margin
        plot_bottom = height - margin

        self.canvas.create_rectangle(plot_left, plot_top, plot_right, plot_bottom, outline=self._colors["frame"])

        self.canvas.create_text(width / 2, height - margin / 3, text=self._x_axis_label, fill=self._colors["text"])
        self.canvas.create_text(margin / 3, height / 2, text=self._y_axis_label, angle=90, fill=self._colors["text"])

        if not self._data:
            self.canvas.create_text(width / 2, height / 2, text="尚未开始测量", fill=self._colors["placeholder"])
            return

        min_x, max_x = self._x_range
        min_y, max_y = self._y_range
        span_x = max(max_x - min_x, 1e-12)
        span_y = max(max_y - min_y, 1e-12)

        x_ticks = self._generate_ticks(min_x, max_x)
        y_ticks = self._generate_ticks(min_y, max_y)

        for value, label in x_ticks:
            position = plot_left + ((value - min_x) / span_x) * (plot_right - plot_left)
            self.canvas.create_line(position, plot_top, position, plot_bottom, fill=self._colors["grid"])
            self.canvas.create_text(
                position,
                plot_bottom + 12,
                text=label,
                anchor="n",
                font=("Arial", 8),
                fill=self._colors["text"],
            )

        for value, label in y_ticks:
            position = plot_bottom - ((value - min_y) / span_y) * (plot_bottom - plot_top)
            self.canvas.create_line(plot_left, position, plot_right, position, fill=self._colors["grid"])
            self.canvas.create_text(
                plot_left - 6,
                position,
                text=label,
                anchor="e",
                font=("Arial", 8),
                fill=self._colors["text"],
            )

        coordinates = []
        for x_value, y_value in self._data:
            x = plot_left + ((x_value - min_x) / span_x) * (plot_right - plot_left)
            y = plot_bottom - ((y_value - min_y) / span_y) * (plot_bottom - plot_top)
            coordinates.append((x, y))

        for i in range(1, len(coordinates)):
            x0, y0 = coordinates[i - 1]
            x1, y1 = coordinates[i]
            self.canvas.create_line(x0, y0, x1, y1, fill=self._colors["line"], width=2)

        for x, y in coordinates:
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=self._colors["point"], outline="")

    def _palette_for_theme(self, theme: str) -> Dict[str, str]:
        if theme == "dark":
            return {
                "background": "#1f1f1f",
                "frame": "#4a4a4a",
                "grid": "#333333",
                "line": "#4c8edb",
                "point": "#4c8edb",
                "text": "#f0f0f0",
                "placeholder": "#aaaaaa",
            }
        return {
            "background": "#ffffff",
            "frame": "#a0a0a0",
            "grid": "#f0f0f0",
            "line": "#0066cc",
            "point": "#0066cc",
            "text": "#202020",
            "placeholder": "#888888",
        }

    def apply_theme(self, theme: str) -> None:
        self._theme = theme
        self._colors = self._palette_for_theme(theme)
        self.canvas.configure(background=self._colors["background"])
        self.redraw()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class MeasurementApp(tk.Tk):
    """Main window for the 2636B software replica."""

    def __init__(self) -> None:
        super().__init__()
        self.title("2636B Measurement Software")
        self.geometry("1280x720")
        self.minsize(1150, 650)

        self.instrument: InstrumentSimulator = InstrumentSimulator()
        self._measurement_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.measurements: List[MeasurementPoint] = []

        self.file_path_var = tk.StringVar()
        self.measure_name_var = tk.StringVar(value="默认测试")
        self.sample_name_var = tk.StringVar(value="Sample-001")
        self.operator_var = tk.StringVar(value="Operator")
        self.device_status_var = tk.StringVar(value="未连接")
        self.auto_save_var = tk.BooleanVar(value=False)
        self.summary_var = tk.StringVar(value="准备就绪")
        self.resource_var = tk.StringVar(value="内置模拟器")
        self.resource_info_var = tk.StringVar(value="使用内置模拟器")
        self._resource_map: Dict[str, VisaResourceInfo] = {}
        self._current_setpoints: List[float] = []
        self.x_axis_var = tk.StringVar(value="电压 (V)")
        self.y_axis_var = tk.StringVar(value="电流 (A)")
        self._time_tests: Dict[str, TimeTestContext] = {}
        self.manual_hold_var = tk.BooleanVar(value=False)
        self.manual_output_status = tk.StringVar(value="输出未配置")
        self.manual_measurement_var = tk.StringVar(value="--")
        self.threshold_metric_var = tk.StringVar(value="电流 (A)")
        self.threshold_relation_var = tk.StringVar(value=">=")
        self.threshold_value_var = tk.DoubleVar(value=0.001)
        self.threshold_result_var = tk.StringVar(value="尚未分析")
        self.analysis_source_var = tk.StringVar(value="I/V测量")
        self.breakdown_source_var = tk.StringVar(value="I/V测量")
        self.threshold_summary_var = tk.StringVar(value="--")
        self.breakdown_limit_var = tk.DoubleVar(value=100.0)
        self.breakdown_result_var = tk.StringVar(value="尚未分析")
        self.breakdown_summary_var = tk.StringVar(value="")
        self.breakdown_window_var = tk.IntVar(value=5)
        self.script_output_var = tk.StringVar(value="尚未执行脚本")
        self.script_text: Optional[tk.Text] = None
        self.script_output: Optional[tk.Text] = None
        self.manual_channel_var = tk.StringVar(value="Channel A")
        self.manual_mode_var = tk.StringVar(value="电压源")
        self.manual_level_var = tk.DoubleVar(value=0.0)
        self.manual_compliance_var = tk.DoubleVar(value=10.0)
        self._script_thread: Optional[threading.Thread] = None
        self._script_stop_event = threading.Event()
        self.auto_scan_on_start = tk.BooleanVar(value=False)
        self.confirm_exit_var = tk.BooleanVar(value=True)
        self.theme_var = tk.StringVar(value="light")
        self.max_log_lines = 500
        self.max_log_lines_var = tk.IntVar(value=self.max_log_lines)
        self._config_path = Path.home() / ".2636b_controller.json"

        self._load_preferences()
        self._build_layout()
        self._configure_styles()
        self._apply_theme(self.theme_var.get())
        self.theme_var.trace_add("write", lambda *_: self._apply_theme(self.theme_var.get()))
        self.max_log_lines_var.trace_add("write", lambda *_: self._update_log_limit())
        self.protocol("WM_DELETE_WINDOW", self.on_exit)

        if self.auto_scan_on_start.get():
            self.after(1200, self.scan_instruments)

    # ------------------------------------------------------------------
    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", padding=6)
        style.configure("Primary.TButton", font=("Microsoft YaHei", 12, "bold"), foreground="#ffffff")
        style.configure("Danger.TButton", foreground="#ffffff")

    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        root_pane = ttk.Frame(self)
        root_pane.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(root_pane, width=240)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        self._build_left_panel(left_panel)

        main_panel = ttk.Frame(root_pane)
        main_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_main_panel(main_panel)

        status_bar = ttk.Label(self, textvariable=self.summary_var, anchor="w", relief=tk.SUNKEN, padding=8)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    # ------------------------------------------------------------------
    def _build_left_panel(self, parent: ttk.Frame) -> None:
        # Connection frame
        connection_frame = ttk.LabelFrame(parent, text="设备控制")
        connection_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        scan_btn = ttk.Button(connection_frame, text="扫描设备", command=self.scan_instruments)
        scan_btn.pack(fill=tk.X, pady=(4, 2))

        self.resource_combo = ttk.Combobox(connection_frame, textvariable=self.resource_var, state="readonly")
        self.resource_combo.pack(fill=tk.X, pady=2)
        self.resource_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_resource_selected())

        ttk.Label(connection_frame, textvariable=self.resource_info_var, wraplength=200, foreground="#555555").pack(fill=tk.X, pady=(2, 6))

        connect_btn = ttk.Button(connection_frame, text="连接仪器", command=self.connect_instrument)
        connect_btn.pack(fill=tk.X, pady=2)

        disconnect_btn = ttk.Button(connection_frame, text="断开连接", command=self.disconnect_instrument)
        disconnect_btn.pack(fill=tk.X, pady=2)

        ttk.Label(connection_frame, text="仪器状态:").pack(anchor="w", pady=(6, 0))
        ttk.Label(connection_frame, textvariable=self.device_status_var, foreground="#3073b6", font=("Microsoft YaHei", 11, "bold")).pack(anchor="w")

        self._update_resource_options([])

        # Measurement buttons
        measurement_frame = ttk.LabelFrame(parent, text="测量操作")
        measurement_frame.pack(fill=tk.X, padx=10, pady=(5, 5))

        self.start_btn = ttk.Button(measurement_frame, text="开始", style="Primary.TButton", command=self.start_measurement)
        self.start_btn.pack(fill=tk.X, pady=5)

        self.stop_btn = ttk.Button(measurement_frame, text="停止", style="Danger.TButton", command=self.stop_measurement, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=5)

        save_btn = ttk.Button(measurement_frame, text="保存结果", command=self.save_results)
        save_btn.pack(fill=tk.X, pady=(15, 4))

        export_btn = ttk.Button(measurement_frame, text="导出配置", command=self.export_configuration)
        export_btn.pack(fill=tk.X, pady=4)

        clear_btn = ttk.Button(measurement_frame, text="清空数据", command=self.clear_results)
        clear_btn.pack(fill=tk.X, pady=4)

        # File options
        file_frame = ttk.LabelFrame(parent, text="文件设置")
        file_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        entry = ttk.Entry(file_frame, textvariable=self.file_path_var)
        entry.pack(fill=tk.X, pady=(4, 2))
        browse_btn = ttk.Button(file_frame, text="浏览...", command=self.select_directory)
        browse_btn.pack(fill=tk.X)

        auto_save = ttk.Checkbutton(file_frame, text="自动保存", variable=self.auto_save_var)
        auto_save.pack(anchor="w", pady=5)

        log_frame = ttk.LabelFrame(parent, text="操作记录")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.log_text = tk.Text(log_frame, height=12, state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    def _build_main_panel(self, parent: ttk.Frame) -> None:
        # Notebook tabs replicating screenshot layout
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        iv_tab = ttk.Frame(notebook)
        notebook.add(iv_tab, text="I/V测量")
        self._build_iv_tab(iv_tab)

        vt_tab = ttk.Frame(notebook)
        notebook.add(vt_tab, text="V/t测试")
        self._build_time_tab(
            vt_tab,
            key="V/t测试",
            name="电压-时间记录",
            required_mode="电流源",
            y_axis="电压 (V)",
            target_field="voltage",
            default_level=0.001,
        )

        it_tab = ttk.Frame(notebook)
        notebook.add(it_tab, text="I/t测试")
        self._build_time_tab(
            it_tab,
            key="I/t测试",
            name="电流-时间记录",
            required_mode="电压源",
            y_axis="电流 (A)",
            target_field="current",
            default_level=1.0,
        )

        threshold_tab = ttk.Frame(notebook)
        notebook.add(threshold_tab, text="阈值扫描")
        self._build_threshold_tab(threshold_tab)

        breakdown_tab = ttk.Frame(notebook)
        notebook.add(breakdown_tab, text="击穿测试")
        self._build_breakdown_tab(breakdown_tab)

        output_tab = ttk.Frame(notebook)
        notebook.add(output_tab, text="输出控制")
        self._build_output_tab(output_tab)

        script_tab = ttk.Frame(notebook)
        notebook.add(script_tab, text="脚本执行")
        self._build_script_tab(script_tab)

        system_tab = ttk.Frame(notebook)
        notebook.add(system_tab, text="系统设置")
        self._build_system_tab(system_tab)

    # ------------------------------------------------------------------
    def _build_iv_tab(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent)
        header.pack(fill=tk.X, padx=10, pady=(10, 5))

        ttk.Label(header, text="测试名称:").pack(side=tk.LEFT)
        ttk.Entry(header, width=16, textvariable=self.measure_name_var).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(header, text="样品编号:").pack(side=tk.LEFT)
        ttk.Entry(header, width=16, textvariable=self.sample_name_var).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(header, text="操作者:").pack(side=tk.LEFT)
        ttk.Entry(header, width=12, textvariable=self.operator_var).pack(side=tk.LEFT, padx=(0, 10))

        timestamp_label = ttk.Label(header, text="测试时间:")
        timestamp_label.pack(side=tk.LEFT)
        self.timestamp_value = ttk.Label(header, text="--")
        self.timestamp_value.pack(side=tk.LEFT)

        graph_frame = ttk.Frame(parent)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))

        axis_bar = ttk.Frame(graph_frame)
        axis_bar.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(axis_bar, text="横轴:").pack(side=tk.LEFT)
        x_combo = ttk.Combobox(axis_bar, textvariable=self.x_axis_var, state="readonly", width=14)
        x_combo.pack(side=tk.LEFT, padx=(2, 12))
        ttk.Label(axis_bar, text="纵轴:").pack(side=tk.LEFT)
        y_combo = ttk.Combobox(axis_bar, textvariable=self.y_axis_var, state="readonly", width=14)
        y_combo.pack(side=tk.LEFT, padx=(2, 12))

        self.graph = SweepGraph(graph_frame)
        x_combo.configure(values=self.graph.axis_choices)
        y_combo.configure(values=self.graph.axis_choices)
        self.graph.pack(fill=tk.BOTH, expand=True)
        x_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_graph())
        y_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_graph())
        self._refresh_graph()

        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)

        self._build_sweep_settings(controls_frame)
        self._build_result_table(parent)

    # ------------------------------------------------------------------
    def _build_sweep_settings(self, parent: ttk.Frame) -> None:
        sweep_frame = ttk.LabelFrame(parent, text="源表参数")
        sweep_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        entries = [
            ("通道", "channel", ["Channel A", "Channel B"]),
            ("工作模式", "mode", ["电流源", "电压源"]),
        ]
        for idx, (label, field_name, values) in enumerate(entries):
            ttk.Label(sweep_frame, text=label + ":").grid(row=idx, column=0, sticky="w", padx=6, pady=3)
            combo = ttk.Combobox(sweep_frame, values=values, state="readonly")
            combo.grid(row=idx, column=1, sticky="ew", padx=6, pady=3)
            combo.current(values.index(getattr(self.instrument.settings, field_name)))
            combo.bind("<<ComboboxSelected>>", lambda e, name=field_name, widget=combo: self._update_setting(name, widget.get()))

        autorange = ttk.Checkbutton(
            sweep_frame,
            text="自动量程",
            command=lambda: self._update_setting("autorange", bool(autorange_var.get())),
            variable=(autorange_var := tk.IntVar(value=1 if self.instrument.settings.autorange else 0)),
        )
        autorange.grid(row=2, column=0, sticky="w", padx=6, pady=3)

        nplc_label = ttk.Label(sweep_frame, text="积分时间(NPLC):")
        nplc_label.grid(row=2, column=1, sticky="e", padx=6, pady=3)
        nplc_entry = ttk.Entry(sweep_frame)
        nplc_entry.insert(0, str(self.instrument.settings.nplc))
        nplc_entry.grid(row=2, column=2, sticky="ew", padx=6, pady=3)
        nplc_entry.bind("<FocusOut>", lambda e: self._safe_float_update("nplc", nplc_entry.get(), default=1.0))

        ttk.Label(sweep_frame, text="触发延时(ms):").grid(row=3, column=0, sticky="w", padx=6, pady=3)
        delay_entry = ttk.Entry(sweep_frame)
        delay_entry.insert(0, str(self.instrument.settings.trigger_delay_ms))
        delay_entry.grid(row=3, column=1, sticky="ew", padx=6, pady=3)
        delay_entry.bind("<FocusOut>", lambda e: self._safe_float_update("trigger_delay_ms", delay_entry.get(), default=10.0))

        ttk.Label(sweep_frame, text="合规电压(V):").grid(row=3, column=2, sticky="w", padx=6, pady=3)
        compliance_entry = ttk.Entry(sweep_frame)
        compliance_entry.insert(0, str(self.instrument.settings.compliance_voltage))
        compliance_entry.grid(row=3, column=3, sticky="ew", padx=6, pady=3)
        compliance_entry.bind("<FocusOut>", lambda e: self._safe_float_update("compliance_voltage", compliance_entry.get(), default=10.0))

        sweep_frame.columnconfigure(1, weight=1)
        sweep_frame.columnconfigure(2, weight=1)
        sweep_frame.columnconfigure(3, weight=1)

        sweep_range = ttk.LabelFrame(parent, text="扫描参数")
        sweep_range.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        labels = ["起始值", "终止值", "步进", "点数"]
        fields = ["start_level", "stop_level", "step", "sweep_points"]
        defaults = [self.instrument.settings.start_level, self.instrument.settings.stop_level, self.instrument.settings.step, self.instrument.settings.sweep_points]
        for idx, (label, field, default) in enumerate(zip(labels, fields, defaults)):
            ttk.Label(sweep_range, text=f"{label}:").grid(row=idx, column=0, sticky="w", padx=6, pady=4)
            entry = ttk.Entry(sweep_range)
            entry.insert(0, str(default))
            entry.grid(row=idx, column=1, sticky="ew", padx=6, pady=4)
            entry.bind("<FocusOut>", lambda e, field=field, widget=entry: self._update_numeric_setting(field, widget.get()))
        sweep_range.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------
    def _update_setting(self, field_name: str, value) -> None:
        setattr(self.instrument.settings, field_name, value)
        self._log(f"更新参数 {field_name} -> {value}")

    def _safe_float_update(self, field_name: str, value: str, *, default: float) -> None:
        try:
            numeric = float(value)
        except ValueError:
            messagebox.showerror("无效输入", f"请输入有效数字: {value!r}")
            numeric = default
        setattr(self.instrument.settings, field_name, numeric)
        self._log(f"更新参数 {field_name} -> {numeric}")

    def _update_numeric_setting(self, field_name: str, value: str) -> None:
        try:
            numeric = float(value) if field_name != "sweep_points" else int(value)
        except ValueError:
            messagebox.showerror("无效输入", f"请输入有效数字: {value!r}")
            return
        setattr(self.instrument.settings, field_name, numeric)
        self._log(f"更新参数 {field_name} -> {numeric}")

    # ------------------------------------------------------------------
    def _build_result_table(self, parent: ttk.Frame) -> None:
        table_frame = ttk.LabelFrame(parent, text="测量结果")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        columns = ("序号", "时间", "电压(V)", "电流(A)", "电阻(Ω)", "功率(W)")
        self.result_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
        for column in columns:
            self.result_tree.heading(column, text=column)
            self.result_tree.column(column, anchor="center", width=120)
        self.result_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.result_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_tree.configure(yscrollcommand=scrollbar.set)

    # ------------------------------------------------------------------
    def _build_time_tab(
        self,
        parent: ttk.Frame,
        *,
        key: str,
        name: str,
        required_mode: str,
        y_axis: str,
        target_field: str,
        default_level: float,
    ) -> None:
        description = ttk.Label(
            parent,
            text=f"{name} - 固定源输出下连续采集，记录仪器电压、电流随时间的变化。",
            foreground="#555555",
            wraplength=880,
            anchor="w",
            justify=tk.LEFT,
        )
        description.pack(fill=tk.X, padx=12, pady=(12, 4))

        config = ttk.LabelFrame(parent, text="采集设置")
        config.pack(fill=tk.X, padx=12, pady=6)

        duration_var = tk.DoubleVar(value=30.0)
        interval_var = tk.DoubleVar(value=0.5)
        level_var = tk.DoubleVar(value=default_level)
        status_var = tk.StringVar(value="准备就绪")
        metrics_var = tk.StringVar(value="--")

        ttk.Label(config, text="采集时长(s):").grid(row=0, column=0, padx=6, pady=4, sticky="w")
        ttk.Entry(config, width=10, textvariable=duration_var).grid(row=0, column=1, padx=6, pady=4, sticky="ew")

        ttk.Label(config, text="采样间隔(s):").grid(row=0, column=2, padx=6, pady=4, sticky="w")
        ttk.Entry(config, width=10, textvariable=interval_var).grid(row=0, column=3, padx=6, pady=4, sticky="ew")

        ttk.Label(config, text="源输出设定:").grid(row=0, column=4, padx=6, pady=4, sticky="w")
        ttk.Entry(config, width=12, textvariable=level_var).grid(row=0, column=5, padx=6, pady=4, sticky="ew")

        ttk.Label(config, text=f"推荐模式: {required_mode}", foreground="#666666").grid(
            row=0, column=6, padx=6, pady=4, sticky="w"
        )

        start_btn = ttk.Button(config, text="开始采集", style="Primary.TButton", command=lambda: self._start_time_test(key))
        start_btn.grid(row=1, column=0, columnspan=2, padx=6, pady=6, sticky="ew")
        stop_btn = ttk.Button(
            config,
            text="停止",
            style="Danger.TButton",
            state=tk.DISABLED,
            command=lambda: self._stop_time_test(key),
        )
        stop_btn.grid(row=1, column=2, columnspan=1, padx=6, pady=6, sticky="ew")
        ttk.Button(config, text="清空", command=lambda: self._clear_time_test(key)).grid(
            row=1, column=3, padx=6, pady=6, sticky="ew"
        )
        ttk.Button(config, text="导出数据", command=lambda: self._export_time_series(key)).grid(
            row=1, column=4, padx=6, pady=6, sticky="ew"
        )
        ttk.Label(config, textvariable=status_var).grid(row=1, column=5, columnspan=2, padx=6, pady=6, sticky="w")

        config.columnconfigure(1, weight=1)
        config.columnconfigure(3, weight=1)
        config.columnconfigure(5, weight=1)

        graph_frame = ttk.LabelFrame(parent, text="实时曲线")
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 6))

        axis_bar = ttk.Frame(graph_frame)
        axis_bar.pack(fill=tk.X, padx=8, pady=(6, 4))
        x_axis_var = tk.StringVar(value="时间 (s)")
        y_axis_var = tk.StringVar(value=y_axis)
        ttk.Label(axis_bar, text="横轴:").pack(side=tk.LEFT)
        x_combo = ttk.Combobox(axis_bar, width=16, state="readonly", textvariable=x_axis_var)
        x_combo.pack(side=tk.LEFT, padx=(2, 12))
        ttk.Label(axis_bar, text="纵轴:").pack(side=tk.LEFT)
        y_combo = ttk.Combobox(axis_bar, width=16, state="readonly", textvariable=y_axis_var)
        y_combo.pack(side=tk.LEFT, padx=(2, 12))

        graph = SweepGraph(graph_frame)
        graph.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        x_combo.configure(values=graph.axis_choices)
        y_combo.configure(values=graph.axis_choices)
        x_combo.bind("<<ComboboxSelected>>", lambda _e, name=key: self._refresh_time_graph(name))
        y_combo.bind("<<ComboboxSelected>>", lambda _e, name=key: self._refresh_time_graph(name))

        result_frame = ttk.LabelFrame(parent, text="数据记录")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))
        columns = ("序号", "时间", "电压(V)", "电流(A)", "功率(W)")
        tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=8)
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, anchor="center", width=120)
        tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        scroll = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=tree.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=scroll.set)

        summary = ttk.Frame(parent)
        summary.pack(fill=tk.X, padx=12, pady=(0, 12))
        ttk.Label(summary, text="统计:").pack(side=tk.LEFT)
        ttk.Label(summary, textvariable=metrics_var).pack(side=tk.LEFT, padx=(4, 0))

        context = TimeTestContext(
            name=key,
            required_mode=required_mode,
            y_axis_label=y_axis,
            target_field=target_field,
            graph=graph,
            tree=tree,
            start_button=start_btn,
            stop_button=stop_btn,
            duration_var=duration_var,
            interval_var=interval_var,
            level_var=level_var,
            x_axis_var=x_axis_var,
            y_axis_var=y_axis_var,
            status_var=status_var,
            metrics_var=metrics_var,
        )
        self._time_tests[key] = context
        self._refresh_time_graph(key)

    # ------------------------------------------------------------------
    def _get_time_context(self, key: str) -> Optional[TimeTestContext]:
        return self._time_tests.get(key)

    def _start_time_test(self, key: str) -> None:
        context = self._get_time_context(key)
        if context is None:
            return
        if not self.instrument.is_connected:
            messagebox.showwarning("警告", "请先连接仪器以进行采集")
            return
        if context.thread and context.thread.is_alive():
            messagebox.showinfo("提示", f"{context.name} 正在采集中")
            return
        try:
            duration = float(context.duration_var.get())
            interval = float(context.interval_var.get())
            level = float(context.level_var.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("无效参数", "请正确填写时长、间隔和输出设定。")
            return
        if duration <= 0 or interval <= 0:
            messagebox.showerror("无效参数", "时长和间隔必须为正数")
            return
        total_points = max(int(duration / interval) + 1, 1)
        total_points = min(total_points, 2000)
        context.points.clear()
        for item in context.tree.get_children():
            context.tree.delete(item)
        self._refresh_time_graph(key)
        context.status_var.set("采集中...")
        context.metrics_var.set("--")
        context.start_button.state(["disabled"])
        context.stop_button.state(["!disabled"])
        context.stop_event.clear()
        self.instrument.settings.mode = context.required_mode
        self.instrument.settings.start_level = level
        self.instrument.settings.stop_level = level
        self.instrument.settings.step = 0.0
        thread = threading.Thread(
            target=self._run_time_test,
            args=(key, total_points, interval, level),
            daemon=True,
        )
        context.thread = thread
        self._log(f"{context.name} 开始采集，共 {total_points} 点，间隔 {interval:.3g}s")
        thread.start()

    def _stop_time_test(self, key: str) -> None:
        context = self._get_time_context(key)
        if context is None:
            return
        context.stop_event.set()
        if context.thread and context.thread.is_alive():
            context.thread.join(timeout=0.5)
        context.start_button.state(["!disabled"])
        context.stop_button.state(["disabled"])
        context.status_var.set("采集已停止")
        self._log(f"{context.name} 已停止")

    def _run_time_test(self, key: str, total_points: int, interval: float, level: float) -> None:
        context = self._get_time_context(key)
        if context is None:
            return
        setpoints = [level] * max(total_points, 1)
        try:
            self.instrument.prepare_measurement(setpoints)
        except Exception as exc:
            self.after(0, self._time_test_failed, key, exc)
            return
        base_time = _dt.datetime.now()
        try:
            for index in range(1, total_points + 1):
                if context.stop_event.is_set():
                    break
                jitter = (index - 1) * 1e-6
                raw_point = self.instrument.generate_point(index, level + jitter)
                timestamp = base_time + _dt.timedelta(seconds=(index - 1) * interval)
                point = MeasurementPoint(
                    index=index,
                    timestamp=timestamp,
                    voltage=raw_point.voltage,
                    current=raw_point.current,
                )
                context.points.append(point)
                self.after(0, self._append_time_point, key, point)
                time.sleep(max(interval, 0.02))
        except Exception as exc:
            self.after(0, self._time_test_failed, key, exc)
        else:
            self.after(0, self._time_test_finished, key, not context.stop_event.is_set())
        finally:
            try:
                self.instrument.finalize_measurement()
            except Exception:
                pass
            context.thread = None

    def _append_time_point(self, key: str, point: MeasurementPoint) -> None:
        context = self._get_time_context(key)
        if context is None:
            return
        context.tree.insert(
            "",
            tk.END,
            values=(
                point.index,
                point.timestamp.strftime("%H:%M:%S"),
                f"{point.voltage:.6f}",
                f"{point.current:.6f}",
                f"{point.power:.6f}",
            ),
        )
        context.status_var.set(f"采集点数: {len(context.points)}")
        self._update_time_metrics(key)
        self._refresh_time_graph(key)

    def _time_test_finished(self, key: str, completed: bool) -> None:
        context = self._get_time_context(key)
        if context is None:
            return
        context.start_button.state(["!disabled"])
        context.stop_button.state(["disabled"])
        context.status_var.set("采集完成" if completed else "采集已停止")
        self._log(f"{context.name} {'完成' if completed else '停止'}")
        self._update_time_metrics(key)

    def _time_test_failed(self, key: str, error: Exception) -> None:
        context = self._get_time_context(key)
        if context is None:
            return
        context.start_button.state(["!disabled"])
        context.stop_button.state(["disabled"])
        context.status_var.set("采集失败")
        messagebox.showerror("采集失败", f"{context.name} 执行失败: {error}")
        self._log(f"{context.name} 采集失败: {error}")

    def _refresh_time_graph(self, key: str) -> None:
        context = self._get_time_context(key)
        if context is None:
            return
        context.graph.set_data(
            context.points,
            x_axis=context.x_axis_var.get(),
            y_axis=context.y_axis_var.get(),
        )

    def _update_time_metrics(self, key: str) -> None:
        context = self._get_time_context(key)
        if context is None:
            return
        if not context.points:
            context.metrics_var.set("--")
            return
        values = [getattr(point, context.target_field) for point in context.points]
        mean_value = statistics.fmean(values)
        context.metrics_var.set(
            f"均值 {mean_value:.5g} | 最小 {min(values):.5g} | 最大 {max(values):.5g}"
        )

    def _clear_time_test(self, key: str) -> None:
        context = self._get_time_context(key)
        if context is None:
            return
        if context.thread and context.thread.is_alive():
            messagebox.showinfo("提示", "请先停止采集")
            return
        context.points.clear()
        for item in context.tree.get_children():
            context.tree.delete(item)
        context.status_var.set("数据已清空")
        context.metrics_var.set("--")
        self._refresh_time_graph(key)
        self._log(f"{context.name} 数据清空")

    def _export_time_series(self, key: str) -> None:
        context = self._get_time_context(key)
        if context is None:
            return
        if not context.points:
            messagebox.showinfo("提示", "当前没有可导出的数据")
            return
        filetypes = [("CSV 文件", "*.csv")]
        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=filetypes)
        if not filename:
            return
        self._write_csv(Path(filename), context.points)
        messagebox.showinfo("提示", "数据导出成功")
        self._log(f"{context.name} 数据导出 -> {filename}")

    # ------------------------------------------------------------------
    def _available_analysis_sources(self) -> List[str]:
        sources = ["I/V测量"]
        sources.extend(self._time_tests.keys())
        return sources

    def _get_dataset_by_source(self, source: str) -> List[MeasurementPoint]:
        if source == "I/V测量":
            return list(self.measurements)
        context = self._get_time_context(source)
        if context:
            return list(context.points)
        return []

    def _build_threshold_tab(self, parent: ttk.Frame) -> None:
        description = ttk.Label(
            parent,
            text="对采集到的数据进行阈值判断，可快速定位导通、饱和或击穿点。",
            wraplength=900,
            foreground="#555555",
            anchor="w",
            justify=tk.LEFT,
        )
        description.pack(fill=tk.X, padx=12, pady=(12, 4))

        config = ttk.LabelFrame(parent, text="分析参数")
        config.pack(fill=tk.X, padx=12, pady=6)

        ttk.Label(config, text="数据来源:").grid(row=0, column=0, padx=6, pady=4, sticky="w")
        source_combo = ttk.Combobox(config, textvariable=self.analysis_source_var, state="readonly", width=18)
        sources = self._available_analysis_sources()
        source_combo.configure(values=sources)
        if self.analysis_source_var.get() not in sources:
            self.analysis_source_var.set(sources[0])
        source_combo.grid(row=0, column=1, padx=6, pady=4, sticky="w")

        ttk.Label(config, text="监测量:").grid(row=0, column=2, padx=6, pady=4, sticky="w")
        metric_choices = [label for label in SweepGraph.AXIS_FIELDS if label != "序号"]
        metric_combo = ttk.Combobox(
            config,
            textvariable=self.threshold_metric_var,
            state="readonly",
            width=18,
            values=metric_choices,
        )
        metric_combo.grid(row=0, column=3, padx=6, pady=4, sticky="w")

        ttk.Label(config, text="阈值:").grid(row=0, column=4, padx=6, pady=4, sticky="w")
        ttk.Entry(config, width=12, textvariable=self.threshold_value_var).grid(row=0, column=5, padx=6, pady=4)

        relation_frame = ttk.Frame(config)
        relation_frame.grid(row=0, column=6, padx=6, pady=4, sticky="w")
        ttk.Radiobutton(relation_frame, text="≥", value=">=", variable=self.threshold_relation_var).pack(side=tk.LEFT)
        ttk.Radiobutton(relation_frame, text="≤", value="<=", variable=self.threshold_relation_var).pack(side=tk.LEFT)

        ttk.Button(config, text="分析阈值", command=self._analyze_threshold).grid(row=0, column=7, padx=6, pady=4)
        ttk.Button(config, text="导出报告", command=self._export_threshold_report).grid(row=0, column=8, padx=6, pady=4)

        config.columnconfigure(1, weight=1)
        config.columnconfigure(3, weight=1)
        config.columnconfigure(5, weight=1)

        result_frame = ttk.LabelFrame(parent, text="分析结果")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))
        ttk.Label(result_frame, textvariable=self.threshold_result_var, wraplength=880, justify=tk.LEFT).pack(
            fill=tk.X, padx=10, pady=(10, 6)
        )
        ttk.Label(result_frame, text="数据范围:").pack(anchor="w", padx=10)
        ttk.Label(result_frame, textvariable=self.threshold_summary_var, foreground="#555555").pack(
            anchor="w", padx=16, pady=(0, 10)
        )

    def _analyze_threshold(self) -> None:
        dataset = self._get_dataset_by_source(self.analysis_source_var.get())
        if not dataset:
            messagebox.showinfo("提示", "当前没有可供分析的数据，请先进行测量。")
            return
        metric_label = self.threshold_metric_var.get()
        field, _unit = SweepGraph.AXIS_FIELDS.get(metric_label, ("current", "A"))
        try:
            threshold = float(self.threshold_value_var.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("无效阈值", "请输入合法的数值阈值。")
            return
        relation = self.threshold_relation_var.get()
        base_time = dataset[0].timestamp

        def value_of(point: MeasurementPoint) -> float:
            if field == "index":
                return float(point.index)
            if field == "time":
                return (point.timestamp - base_time).total_seconds()
            attr = getattr(point, field, None)
            return float(attr) if attr is not None else float("nan")

        match: Optional[Tuple[MeasurementPoint, float]] = None
        for point in dataset:
            value = value_of(point)
            if relation == ">=" and value >= threshold:
                match = (point, value)
                break
            if relation == "<=" and value <= threshold:
                match = (point, value)
                break

        values = [value_of(point) for point in dataset]
        finite_values = [v for v in values if math.isfinite(v)]
        if finite_values:
            summary = f"最小 {min(finite_values):.5g} | 最大 {max(finite_values):.5g} | 均值 {statistics.fmean(finite_values):.5g}"
            self.threshold_summary_var.set(summary)
        else:
            self.threshold_summary_var.set("数据不足以计算统计量")

        if match is None:
            self.threshold_result_var.set("未达到阈值，建议扩大测量范围或调整阈值。")
            self._log(f"阈值分析: 未达到阈值 {threshold:.5g} ({metric_label} {relation})")
            return

        point, value = match
        message = (
            f"在第 {point.index} 点达到阈值 ({metric_label} {relation} {threshold:.5g})，"
            f"实际值 {value:.5g}。\n电压 {point.voltage:.5g} V | 电流 {point.current:.5g} A | 时间 {point.timestamp.strftime('%H:%M:%S')}"
        )
        self.threshold_result_var.set(message)
        self._log(f"阈值分析命中: 点 {point.index} -> {value:.5g}")

    def _export_threshold_report(self) -> None:
        if self.threshold_result_var.get() == "尚未分析":
            messagebox.showinfo("提示", "请先执行阈值分析。")
            return
        dataset = self._get_dataset_by_source(self.analysis_source_var.get())
        if not dataset:
            messagebox.showinfo("提示", "当前没有分析数据。")
            return
        report = {
            "source": self.analysis_source_var.get(),
            "metric": self.threshold_metric_var.get(),
            "relation": self.threshold_relation_var.get(),
            "threshold": float(self.threshold_value_var.get()),
            "result": self.threshold_result_var.get(),
            "summary": self.threshold_summary_var.get(),
            "points": len(dataset),
        }
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON 文件", "*.json"), ("文本文件", "*.txt")],
        )
        if not filename:
            return
        with open(filename, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        messagebox.showinfo("提示", "分析报告已导出")
        self._log(f"阈值分析报告导出 -> {filename}")

    def _build_breakdown_tab(self, parent: ttk.Frame) -> None:
        description = ttk.Label(
            parent,
            text="根据斜率变化自动评估器件击穿点，支持窗口平滑与阈值调节。",
            wraplength=900,
            foreground="#555555",
            anchor="w",
            justify=tk.LEFT,
        )
        description.pack(fill=tk.X, padx=12, pady=(12, 4))

        config = ttk.LabelFrame(parent, text="判据设置")
        config.pack(fill=tk.X, padx=12, pady=6)

        ttk.Label(config, text="数据来源:").grid(row=0, column=0, padx=6, pady=4, sticky="w")
        source_combo = ttk.Combobox(config, textvariable=self.breakdown_source_var, state="readonly", width=18)
        sources = self._available_analysis_sources()
        source_combo.configure(values=sources)
        if self.breakdown_source_var.get() not in sources:
            self.breakdown_source_var.set(sources[0])
        source_combo.grid(row=0, column=1, padx=6, pady=4, sticky="w")

        ttk.Label(config, text="斜率阈值(A/V):").grid(row=0, column=2, padx=6, pady=4, sticky="w")
        ttk.Entry(config, width=12, textvariable=self.breakdown_limit_var).grid(row=0, column=3, padx=6, pady=4)

        ttk.Label(config, text="平滑窗口:").grid(row=0, column=4, padx=6, pady=4, sticky="w")
        ttk.Entry(config, width=10, textvariable=self.breakdown_window_var).grid(row=0, column=5, padx=6, pady=4)

        ttk.Button(config, text="执行分析", command=self._analyze_breakdown).grid(row=0, column=6, padx=6, pady=4)

        result = ttk.LabelFrame(parent, text="击穿判定")
        result.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))
        ttk.Label(result, textvariable=self.breakdown_result_var, wraplength=880, justify=tk.LEFT).pack(
            fill=tk.X, padx=10, pady=(10, 6)
        )
        ttk.Label(result, text="统计:").pack(anchor="w", padx=10)
        ttk.Label(result, textvariable=self.breakdown_summary_var, foreground="#555555").pack(
            anchor="w", padx=16, pady=(0, 10)
        )

    def _analyze_breakdown(self) -> None:
        dataset = self._get_dataset_by_source(self.breakdown_source_var.get())
        if len(dataset) < 3:
            messagebox.showinfo("提示", "数据点过少，无法进行击穿分析。")
            return
        try:
            window = max(int(self.breakdown_window_var.get()), 2)
        except (ValueError, tk.TclError):
            messagebox.showerror("无效窗口", "请填写正确的窗口尺寸。")
            return
        try:
            limit = float(self.breakdown_limit_var.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("无效阈值", "请填写正确的斜率阈值。")
            return

        slopes: List[Tuple[int, float]] = []
        for index in range(window - 1, len(dataset)):
            subset = dataset[index - window + 1 : index + 1]
            dv = subset[-1].voltage - subset[0].voltage
            di = subset[-1].current - subset[0].current
            if math.isclose(dv, 0.0, rel_tol=1e-9, abs_tol=1e-12):
                slope = float("inf")
            else:
                slope = di / dv
            slopes.append((index, slope))

        match = None
        for idx, slope in slopes:
            if abs(slope) >= abs(limit):
                match = (idx, slope)
                break

        if not slopes:
            self.breakdown_result_var.set("无法计算斜率，请检查数据。")
            return

        peak_slope = max(slopes, key=lambda item: abs(item[1]))
        self.breakdown_summary_var.set(
            f"最大斜率 {peak_slope[1]:.4g} A/V | 数据点 {len(dataset)}"
        )

        if match is None:
            self.breakdown_result_var.set("未检测到击穿迹象，请增加源强度或降低阈值。")
            self._log("击穿分析: 未发现超阈值斜率")
            return

        index, slope = match
        point = dataset[index]
        self.breakdown_result_var.set(
            f"检测到击穿: 第 {point.index} 点斜率 {slope:.4g} A/V。\n"
            f"电压 {point.voltage:.5g} V | 电流 {point.current:.5g} A"
        )
        self._log(f"击穿分析: 点 {point.index} 斜率 {slope:.4g} A/V")

    # ------------------------------------------------------------------
    def _build_output_tab(self, parent: ttk.Frame) -> None:
        description = ttk.Label(
            parent,
            text="对源表输出进行手动控制，支持立即测量并选择是否保持输出。",
            wraplength=900,
            foreground="#555555",
            anchor="w",
            justify=tk.LEFT,
        )
        description.pack(fill=tk.X, padx=12, pady=(12, 4))

        control = ttk.LabelFrame(parent, text="输出设置")
        control.pack(fill=tk.X, padx=12, pady=6)

        ttk.Label(control, text="通道:").grid(row=0, column=0, padx=6, pady=4, sticky="w")
        channel_combo = ttk.Combobox(
            control,
            textvariable=self.manual_channel_var,
            values=["Channel A", "Channel B"],
            state="readonly",
            width=12,
        )
        channel_combo.grid(row=0, column=1, padx=6, pady=4, sticky="w")

        ttk.Label(control, text="模式:").grid(row=0, column=2, padx=6, pady=4, sticky="w")
        mode_combo = ttk.Combobox(
            control,
            textvariable=self.manual_mode_var,
            values=["电压源", "电流源"],
            state="readonly",
            width=12,
        )
        mode_combo.grid(row=0, column=3, padx=6, pady=4, sticky="w")

        ttk.Label(control, text="输出设定:").grid(row=1, column=0, padx=6, pady=4, sticky="w")
        ttk.Entry(control, width=16, textvariable=self.manual_level_var).grid(row=1, column=1, padx=6, pady=4, sticky="ew")

        ttk.Label(control, text="合规值:").grid(row=1, column=2, padx=6, pady=4, sticky="w")
        ttk.Entry(control, width=16, textvariable=self.manual_compliance_var).grid(row=1, column=3, padx=6, pady=4, sticky="ew")

        ttk.Checkbutton(control, text="保持输出", variable=self.manual_hold_var).grid(row=1, column=4, padx=6, pady=4, sticky="w")

        ttk.Button(control, text="应用输出", command=self._apply_manual_output).grid(
            row=0, column=4, padx=6, pady=4, sticky="ew"
        )

        control.columnconfigure(1, weight=1)
        control.columnconfigure(3, weight=1)

        status = ttk.LabelFrame(parent, text="实时测量")
        status.pack(fill=tk.X, padx=12, pady=(0, 10))
        ttk.Label(status, textvariable=self.manual_output_status).pack(anchor="w", padx=10, pady=(10, 4))
        ttk.Label(status, textvariable=self.manual_measurement_var, foreground="#555555").pack(
            anchor="w", padx=16, pady=(0, 10)
        )

    def _apply_manual_output(self) -> None:
        if not self.instrument.is_connected:
            messagebox.showwarning("提示", "请先连接仪器再调整输出。")
            return
        try:
            level = float(self.manual_level_var.get())
            compliance = float(self.manual_compliance_var.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("无效参数", "输出设定和合规值必须为数字。")
            return

        self.instrument.settings.channel = self.manual_channel_var.get()
        self.instrument.settings.mode = self.manual_mode_var.get()
        self.instrument.settings.start_level = level
        self.instrument.settings.stop_level = level
        self.instrument.settings.step = 0.0
        self.instrument.settings.compliance_voltage = compliance

        try:
            self.instrument.prepare_measurement([level])
            point = self.instrument.generate_point(1, level)
            self.manual_measurement_var.set(
                f"电压 {point.voltage:.5g} V | 电流 {point.current:.5g} A"
            )
            self.manual_output_status.set("输出已更新")
            self._log(
                f"手动输出: {self.instrument.settings.mode} -> {level:.5g} (合规 {compliance:.5g})"
            )
        except Exception as exc:
            self.manual_output_status.set("输出失败")
            messagebox.showerror("输出失败", f"无法更新输出: {exc}")
            self._log(f"手动输出失败: {exc}")
        else:
            if not self.manual_hold_var.get():
                try:
                    self.instrument.finalize_measurement()
                except Exception:
                    pass
            else:
                self.manual_output_status.set("输出保持中")

    def _update_manual_defaults(self) -> None:
        self.manual_channel_var.set(self.instrument.settings.channel)
        self.manual_mode_var.set(self.instrument.settings.mode)
        self.manual_level_var.set(self.instrument.settings.start_level)
        self.manual_compliance_var.set(self.instrument.settings.compliance_voltage)

    # ------------------------------------------------------------------
    def _build_script_tab(self, parent: ttk.Frame) -> None:
        description = ttk.Label(
            parent,
            text="通过脚本组合多段测量流程，支持 MODE/LEVEL/MEASURE/SWEEP/WAIT/LOG 指令。",
            wraplength=900,
            foreground="#555555",
            anchor="w",
            justify=tk.LEFT,
        )
        description.pack(fill=tk.X, padx=12, pady=(12, 4))

        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, padx=12, pady=(0, 4))
        ttk.Button(toolbar, text="打开脚本", command=self._open_script_file).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="保存脚本", command=self._save_script_file).pack(side=tk.LEFT, padx=6)
        ttk.Button(toolbar, text="运行脚本", style="Primary.TButton", command=self._run_script).pack(side=tk.LEFT, padx=6)
        ttk.Button(toolbar, text="停止", style="Danger.TButton", command=self._stop_script).pack(side=tk.LEFT, padx=6)
        ttk.Label(toolbar, textvariable=self.script_output_var).pack(side=tk.RIGHT)

        editor_frame = ttk.Frame(parent)
        editor_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)
        self.script_text = tk.Text(editor_frame, wrap=tk.NONE, height=16)
        self.script_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        script_scroll = ttk.Scrollbar(editor_frame, orient=tk.VERTICAL, command=self.script_text.yview)
        script_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.script_text.configure(yscrollcommand=script_scroll.set)
        if not self.script_text.get("1.0", tk.END).strip():
            default_script = (
                "# 示例脚本\n"
                "MODE 电流源\n"
                "LEVEL 0.001\n"
                "MEASURE 10 0.2\n"
                "WAIT 1.0\n"
                "SWEEP -0.001 0.001 0.0005\n"
                "LOG 完成一次扫描\n"
            )
            self.script_text.insert("1.0", default_script)

        output_frame = ttk.LabelFrame(parent, text="脚本输出")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))
        self.script_output = tk.Text(output_frame, height=8, state=tk.DISABLED, wrap=tk.WORD)
        self.script_output.pack(fill=tk.BOTH, expand=True)

    def _open_script_file(self) -> None:
        filename = filedialog.askopenfilename(
            filetypes=[("脚本文件", "*.tsp"), ("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        if not filename:
            return
        try:
            with open(filename, "r", encoding="utf-8") as fh:
                content = fh.read()
        except Exception as exc:
            messagebox.showerror("读取失败", f"无法打开脚本: {exc}")
            return
        if self.script_text is not None:
            self.script_text.delete("1.0", tk.END)
            self.script_text.insert("1.0", content)
        self._log(f"载入脚本 {filename}")

    def _save_script_file(self) -> None:
        if self.script_text is None:
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".tsp",
            filetypes=[("脚本文件", "*.tsp"), ("文本文件", "*.txt"), ("所有文件", "*.*")],
        )
        if not filename:
            return
        try:
            with open(filename, "w", encoding="utf-8") as fh:
                fh.write(self.script_text.get("1.0", tk.END))
        except Exception as exc:
            messagebox.showerror("保存失败", f"无法保存脚本: {exc}")
            return
        self._log(f"脚本已保存到 {filename}")

    def _run_script(self) -> None:
        if self.script_text is None:
            return
        if not self.instrument.is_connected:
            messagebox.showwarning("提示", "脚本执行前请先连接仪器。")
            return
        if self._script_thread and self._script_thread.is_alive():
            messagebox.showinfo("提示", "脚本正在执行中")
            return
        script = self.script_text.get("1.0", tk.END).strip()
        if not script:
            messagebox.showinfo("提示", "脚本内容为空")
            return
        self._script_stop_event.clear()
        self._script_log("开始执行脚本...")
        self._log("开始执行脚本任务")
        self._script_thread = threading.Thread(target=self._script_worker, args=(script,), daemon=True)
        self._script_thread.start()

    def _stop_script(self) -> None:
        if self._script_thread and self._script_thread.is_alive():
            self._script_stop_event.set()
            self._script_log("正在请求停止脚本...")

    def _script_worker(self, script: str) -> None:
        lines = script.splitlines()
        current_level = float(self.manual_level_var.get())
        mode = self.manual_mode_var.get() or self.instrument.settings.mode
        try:
            for raw_line in lines:
                if self._script_stop_event.is_set():
                    break
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                command = parts[0].upper()
                if command == "MODE" and len(parts) >= 2:
                    token = parts[1]
                    if token in ("I", "CURR", "CURRENT", "电流源"):
                        mode = "电流源"
                    else:
                        mode = "电压源"
                    self.instrument.settings.mode = mode
                    self.after(0, self._script_log, f"设置模式 -> {mode}")
                elif command == "LEVEL" and len(parts) >= 2:
                    current_level = float(parts[1])
                    self.instrument.settings.start_level = current_level
                    self.instrument.settings.stop_level = current_level
                    self.after(0, self._script_log, f"设置输出 -> {current_level:.5g}")
                elif command == "WAIT" and len(parts) >= 2:
                    delay = max(float(parts[1]), 0.0)
                    self.after(0, self._script_log, f"等待 {delay:.2f}s")
                    elapsed = 0.0
                    while elapsed < delay and not self._script_stop_event.is_set():
                        step = min(0.2, delay - elapsed)
                        time.sleep(step)
                        elapsed += step
                elif command == "MEASURE":
                    count = int(parts[1]) if len(parts) > 1 else 1
                    dwell = float(parts[2]) if len(parts) > 2 else 0.2
                    self.instrument.settings.mode = mode
                    try:
                        self.instrument.prepare_measurement([current_level] * max(count, 1))
                        for idx in range(1, count + 1):
                            if self._script_stop_event.is_set():
                                break
                            point = self.instrument.generate_point(idx, current_level + idx * 1e-6)
                            self.after(
                                0,
                                self._script_log,
                                f"MEASURE {idx}: V={point.voltage:.5g} V I={point.current:.5g} A",
                            )
                            time.sleep(max(dwell, 0.02))
                    finally:
                        if not self.manual_hold_var.get():
                            try:
                                self.instrument.finalize_measurement()
                            except Exception:
                                pass
                elif command == "SWEEP" and len(parts) >= 4:
                    start = float(parts[1])
                    stop = float(parts[2])
                    step = float(parts[3])
                    if math.isclose(step, 0.0, rel_tol=1e-9, abs_tol=1e-12):
                        if math.isclose(stop, start, rel_tol=1e-9, abs_tol=1e-12):
                            values = [start]
                        else:
                            step = (stop - start) / max(len(self._current_setpoints) or 1, 1)
                            if math.isclose(step, 0.0, rel_tol=1e-9, abs_tol=1e-12):
                                step = stop - start
                    direction = 1 if stop >= start else -1
                    if direction > 0 and step < 0:
                        step = abs(step)
                    if direction < 0 and step > 0:
                        step = -abs(step)
                    if not locals().get("values"):
                        values = []
                        current = start
                        for _ in range(2000):
                            values.append(current)
                            current += step
                            if (direction > 0 and current > stop + abs(step) * 0.5) or (
                                direction < 0 and current < stop - abs(step) * 0.5
                            ):
                                break
                        if not values or not math.isclose(values[-1], stop, rel_tol=1e-9, abs_tol=1e-12):
                            values.append(stop)
                    self.instrument.settings.mode = mode
                    try:
                        self.instrument.prepare_measurement(values)
                        for idx, level in enumerate(values, start=1):
                            if self._script_stop_event.is_set():
                                break
                            point = self.instrument.generate_point(idx, level)
                            self.after(
                                0,
                                self._script_log,
                                f"SWEEP {idx}: level={level:.5g} -> V={point.voltage:.5g} I={point.current:.5g}",
                            )
                            time.sleep(0.05)
                    finally:
                        if not self.manual_hold_var.get():
                            try:
                                self.instrument.finalize_measurement()
                            except Exception:
                                pass
                elif command == "LOG":
                    message = " ".join(parts[1:])
                    self.after(0, self._script_log, message or "")
                else:
                    self.after(0, self._script_log, f"未知指令: {line}")
        except Exception as exc:
            self.after(0, self._script_log, f"脚本执行失败: {exc}")
            self.after(0, self._script_finished, False)
            return
        self.after(0, self._script_finished, not self._script_stop_event.is_set())

    def _script_log(self, message: str) -> None:
        timestamp = _dt.datetime.now().strftime("%H:%M:%S")
        self.script_output_var.set(message)
        if self.script_output is not None:
            self.script_output.configure(state=tk.NORMAL)
            self.script_output.insert(tk.END, f"[{timestamp}] {message}\n")
            self.script_output.see(tk.END)
            self.script_output.configure(state=tk.DISABLED)

    def _script_finished(self, success: bool) -> None:
        status = "脚本执行完成" if success else "脚本已停止"
        self.script_output_var.set(status)
        if self._script_thread and not self._script_thread.is_alive():
            self._script_thread = None
        self._log(status)

    # ------------------------------------------------------------------
    def _build_system_tab(self, parent: ttk.Frame) -> None:
        general = ttk.LabelFrame(parent, text="常规设置")
        general.pack(fill=tk.X, padx=12, pady=(12, 6))
        ttk.Checkbutton(general, text="启动时自动扫描设备", variable=self.auto_scan_on_start).grid(
            row=0, column=0, padx=6, pady=6, sticky="w"
        )
        ttk.Checkbutton(general, text="退出前提示确认", variable=self.confirm_exit_var).grid(
            row=0, column=1, padx=6, pady=6, sticky="w"
        )
        ttk.Checkbutton(general, text="测量完成后自动保存", variable=self.auto_save_var).grid(
            row=0, column=2, padx=6, pady=6, sticky="w"
        )

        theme = ttk.LabelFrame(parent, text="界面主题")
        theme.pack(fill=tk.X, padx=12, pady=6)
        ttk.Radiobutton(theme, text="浅色", value="light", variable=self.theme_var).pack(side=tk.LEFT, padx=10, pady=6)
        ttk.Radiobutton(theme, text="深色", value="dark", variable=self.theme_var).pack(side=tk.LEFT, padx=10, pady=6)

        log_frame = ttk.LabelFrame(parent, text="日志管理")
        log_frame.pack(fill=tk.X, padx=12, pady=6)
        ttk.Label(log_frame, text="最大保留行数:").grid(row=0, column=0, padx=6, pady=6, sticky="w")
        ttk.Entry(log_frame, width=10, textvariable=self.max_log_lines_var).grid(row=0, column=1, padx=6, pady=6, sticky="w")
        ttk.Button(log_frame, text="清空日志", command=self._clear_log).grid(row=0, column=2, padx=6, pady=6)

        path_frame = ttk.LabelFrame(parent, text="文件与布局")
        path_frame.pack(fill=tk.X, padx=12, pady=(6, 10))
        ttk.Label(path_frame, text="默认保存目录:").grid(row=0, column=0, padx=6, pady=6, sticky="w")
        entry = ttk.Entry(path_frame, textvariable=self.file_path_var, width=50)
        entry.grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        ttk.Button(path_frame, text="选择", command=self.select_directory).grid(row=0, column=2, padx=6, pady=6)
        ttk.Button(path_frame, text="重置窗口布局", command=self._reset_layout).grid(row=0, column=3, padx=6, pady=6)
        path_frame.columnconfigure(1, weight=1)

    def _reset_layout(self) -> None:
        self.geometry("1280x720")
        self.summary_var.set("已恢复默认窗口大小")

    # ------------------------------------------------------------------
    def _current_theme_colors(self) -> Dict[str, str]:
        if self.theme_var.get() == "dark":
            return {
                "background": "#1f1f1f",
                "panel": "#2b2b2b",
                "panel_active": "#3a3f4b",
                "foreground": "#f0f0f0",
                "primary": "#4c8edb",
                "primary_active": "#3a6fb1",
                "primary_disabled": "#2b4668",
                "danger": "#e06666",
                "danger_active": "#c45454",
                "on_primary": "#ffffff",
                "on_danger": "#ffffff",
                "text_bg": "#1f1f1f",
                "text_fg": "#f0f0f0",
                "grid": "#333333",
            }
        return {
            "background": "#f5f5f5",
            "panel": "#ffffff",
            "panel_active": "#e6eef8",
            "foreground": "#1d1d1d",
            "primary": "#3073b6",
            "primary_active": "#1e5590",
            "primary_disabled": "#9bb7d7",
            "danger": "#d9534f",
            "danger_active": "#c9302c",
            "on_primary": "#ffffff",
            "on_danger": "#ffffff",
            "text_bg": "#ffffff",
            "text_fg": "#1d1d1d",
            "grid": "#e0e0e0",
        }

    def _apply_theme(self, theme: str) -> None:
        colors = self._current_theme_colors()
        style = ttk.Style(self)
        style.configure("TFrame", background=colors["background"])
        style.configure("TLabel", background=colors["background"], foreground=colors["foreground"])
        style.configure("TLabelframe", background=colors["background"], foreground=colors["foreground"])
        style.configure("TLabelframe.Label", background=colors["background"], foreground=colors["foreground"])
        style.configure("TNotebook", background=colors["background"])
        style.configure("TNotebook.Tab", background=colors["panel"], foreground=colors["foreground"])
        style.map(
            "TNotebook.Tab",
            background=[("selected", colors["panel_active"]), ("!selected", colors["panel"])],
        )
        style.configure("Treeview", background=colors["panel"], fieldbackground=colors["panel"], foreground=colors["foreground"])
        style.map(
            "Treeview",
            background=[("selected", colors["primary"])],
            foreground=[("selected", colors["on_primary"])],
        )
        style.configure("TButton", background=colors["panel"], foreground=colors["foreground"])
        style.map("TButton", background=[("active", colors["panel_active"])])
        style.configure(
            "Primary.TButton",
            background=colors["primary"],
            foreground=colors["on_primary"],
        )
        style.map(
            "Primary.TButton",
            background=[("active", colors["primary_active"]), ("disabled", colors["primary_disabled"])],
        )
        style.configure(
            "Danger.TButton",
            background=colors["danger"],
            foreground=colors["on_danger"],
        )
        style.map("Danger.TButton", background=[("active", colors["danger_active"])])

        self.configure(background=colors["background"])
        if hasattr(self, "log_text"):
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.configure(background=colors["text_bg"], foreground=colors["text_fg"], insertbackground=colors["text_fg"])
            self.log_text.configure(state=tk.DISABLED)
        if self.script_text is not None:
            self.script_text.configure(background=colors["text_bg"], foreground=colors["text_fg"], insertbackground=colors["text_fg"])
        if self.script_output is not None:
            self.script_output.configure(state=tk.NORMAL)
            self.script_output.configure(background=colors["text_bg"], foreground=colors["text_fg"], insertbackground=colors["text_fg"])
            self.script_output.configure(state=tk.DISABLED)
        self.graph.apply_theme(theme)
        for context in self._time_tests.values():
            context.graph.apply_theme(theme)

    def _update_log_limit(self) -> None:
        try:
            limit = int(self.max_log_lines_var.get())
        except (ValueError, tk.TclError):
            limit = self.max_log_lines
        self.max_log_lines = max(limit, 100)
        self._trim_log()

    def _trim_log(self) -> None:
        if not hasattr(self, "log_text") or self.max_log_lines <= 0:
            return
        total = self.log_text.index("end-1c")
        try:
            total_lines = int(total.split(".")[0])
        except (ValueError, AttributeError):
            return
        if total_lines <= self.max_log_lines:
            return
        cut = total_lines - self.max_log_lines
        self.log_text.delete("1.0", f"{cut + 1}.0")

    def _clear_log(self) -> None:
        if not hasattr(self, "log_text"):
            return
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self.summary_var.set("日志已清空")

    def _stop_all_time_tests(self) -> None:
        for context in self._time_tests.values():
            context.stop_event.set()
            if context.thread and context.thread.is_alive():
                context.thread.join(timeout=0.5)

    def _load_preferences(self) -> None:
        try:
            data = json.loads(self._config_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception:
            return
        self.theme_var.set(data.get("theme", self.theme_var.get()))
        self.auto_scan_on_start.set(data.get("auto_scan", self.auto_scan_on_start.get()))
        self.confirm_exit_var.set(data.get("confirm_exit", self.confirm_exit_var.get()))
        self.auto_save_var.set(data.get("auto_save", self.auto_save_var.get()))
        self.max_log_lines = int(data.get("log_limit", self.max_log_lines))
        self.max_log_lines_var.set(self.max_log_lines)
        default_dir = data.get("default_directory")
        if default_dir:
            self.file_path_var.set(default_dir)
        geometry = data.get("geometry")
        if geometry:
            self.geometry(geometry)
        self.manual_channel_var.set(data.get("manual_channel", self.manual_channel_var.get()))
        self.manual_mode_var.set(data.get("manual_mode", self.manual_mode_var.get()))
        self.manual_level_var.set(data.get("manual_level", self.manual_level_var.get()))
        self.manual_compliance_var.set(data.get("manual_compliance", self.manual_compliance_var.get()))

    def _save_preferences(self) -> None:
        data = {
            "theme": self.theme_var.get(),
            "auto_scan": bool(self.auto_scan_on_start.get()),
            "confirm_exit": bool(self.confirm_exit_var.get()),
            "auto_save": bool(self.auto_save_var.get()),
            "log_limit": self.max_log_lines,
            "default_directory": self.file_path_var.get(),
            "geometry": self.geometry(),
            "manual_channel": self.manual_channel_var.get(),
            "manual_mode": self.manual_mode_var.get(),
            "manual_level": float(self.manual_level_var.get()),
            "manual_compliance": float(self.manual_compliance_var.get()),
        }
        try:
            self._config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Instrument interaction helpers
    # ------------------------------------------------------------------
    def _update_resource_options(self, resources: List[VisaResourceInfo]) -> None:
        values = ["内置模拟器"]
        self._resource_map = {}
        for info in resources:
            base_label = info.display_name()
            candidate = base_label
            suffix = 1
            while candidate in values:
                suffix += 1
                candidate = f"{base_label} #{suffix}"
            values.append(candidate)
            self._resource_map[candidate] = info
        self.resource_combo.configure(values=values)
        if self.resource_var.get() not in values:
            self.resource_var.set(values[0])
        self._on_resource_selected()

    def _on_resource_selected(self) -> None:
        selection = self.resource_var.get()
        if selection in self._resource_map:
            info = self._resource_map[selection]
            manufacturer = info.manufacturer or "未知厂商"
            model = info.model or "未知型号"
            serial = info.serial or "未提供序列号"
            text = f"{manufacturer} {model}\n资源: {info.resource}\n序列号: {serial}"
            self.resource_info_var.set(text)
        else:
            self.resource_info_var.set("使用内置模拟器")

    def scan_instruments(self) -> None:
        if pyvisa is None:
            messagebox.showwarning("未检测到PyVISA", "需要安装 PyVISA 并配置 NI-VISA 才能连接真实仪器。")
            return
        self.summary_var.set("正在扫描仪器...")
        self.update_idletasks()
        resources = discover_visa_resources()
        if not resources:
            self.summary_var.set("未发现VISA设备，已切换至模拟器")
            messagebox.showinfo("扫描结果", "未发现可用的 VISA 设备，仍可使用内置模拟器进行演示。")
            self._log("未发现任何 VISA 设备，继续使用模拟器")
        else:
            self.summary_var.set(f"发现 {len(resources)} 台可用设备")
            self._log(f"扫描到设备: {', '.join(info.display_name() for info in resources)}")
        self._update_resource_options(resources)

    def connect_instrument(self) -> None:
        if self.instrument.is_connected:
            messagebox.showinfo("提示", "仪器已连接")
            return
        selection = self.resource_var.get()
        if selection in self._resource_map:
            info = self._resource_map[selection]
            instrument = VisaInstrument(info)
        else:
            instrument = InstrumentSimulator()
        instrument.settings = self.instrument.settings  # preserve current configuration
        try:
            instrument.connect()
        except Exception as exc:  # pragma: no cover - defensive guard
            messagebox.showerror("错误", f"连接仪器失败: {exc}")
            return
        self.instrument = instrument  # type: ignore[assignment]
        if isinstance(instrument, VisaInstrument):
            status = instrument.info.display_name()
        else:
            status = "模拟器"
        self.device_status_var.set(f"已连接 ({status})")
        self.summary_var.set("仪器连接成功")
        self._log(f"仪器已连接: {status}")
        self._update_manual_defaults()

    def disconnect_instrument(self) -> None:
        if not self.instrument.is_connected:
            messagebox.showinfo("提示", "仪器未连接")
            return
        try:
            self.instrument.disconnect()
        except Exception as exc:  # pragma: no cover - defensive guard
            messagebox.showerror("错误", f"断开仪器失败: {exc}")
            self._log(f"断开仪器失败: {exc}")
            return
        self.device_status_var.set("未连接")
        self.summary_var.set("仪器已断开")
        self._log("仪器连接断开")
        self.manual_output_status.set("输出未配置")

    # ------------------------------------------------------------------
    def start_measurement(self) -> None:
        if not self.instrument.is_connected:
            messagebox.showwarning("警告", "请先连接仪器")
            return
        if self._measurement_thread and self._measurement_thread.is_alive():
            messagebox.showinfo("提示", "测量正在进行中")
            return
        self.summary_var.set("测量进行中...")
        self._stop_event.clear()
        self.measurements.clear()
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        self._refresh_graph()
        self.timestamp_value.config(text=_dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.start_btn.state(["disabled"])
        self.stop_btn.state(["!disabled"])
        self._current_setpoints = self._compute_setpoints()
        if not self._current_setpoints:
            messagebox.showerror("配置错误", "无法根据当前参数计算扫描点，请检查设置。")
            self.summary_var.set("测量初始化失败")
            self.start_btn.state(["!disabled"])
            self.stop_btn.state(["disabled"])
            return
        try:
            self.instrument.prepare_measurement(self._current_setpoints)
        except Exception as exc:  # pragma: no cover - depends on hardware
            messagebox.showerror("错误", f"仪器初始化失败: {exc}")
            self.summary_var.set("仪器初始化失败")
            self.start_btn.state(["!disabled"])
            self.stop_btn.state(["disabled"])
            self._log(f"仪器初始化失败: {exc}")
            self._current_setpoints = []
            return
        self._log(
            f"开始测量，共 {len(self._current_setpoints)} 点，范围 {self._current_setpoints[0]:.4g} -> {self._current_setpoints[-1]:.4g}"
        )
        self._measurement_thread = threading.Thread(target=self._run_measurement, daemon=True)
        self._measurement_thread.start()

    def _run_measurement(self) -> None:
        dwell = max(self.instrument.settings.trigger_delay_ms / 1000.0, 0.05)
        try:
            for index, level in enumerate(self._current_setpoints, start=1):
                if self._stop_event.is_set():
                    break
                point = self.instrument.generate_point(index, level)
                self.measurements.append(point)
                self.after(0, self._append_point, point)
                time.sleep(dwell)
        except Exception as exc:  # pragma: no cover - depends on hardware
            self.after(0, lambda error=exc: self._measurement_error(error))
        else:
            self.after(0, self._measurement_finished)
        finally:
            try:
                self.instrument.finalize_measurement()
            except Exception:
                pass
            self._measurement_thread = None

    def _compute_setpoints(self) -> List[float]:
        settings = self.instrument.settings
        points = max(int(settings.sweep_points), 1)
        start = float(settings.start_level)
        stop = float(settings.stop_level)
        if points <= 1:
            return [start]
        step = float(settings.step)
        if math.isclose(step, 0.0, rel_tol=1e-9, abs_tol=1e-12):
            step = (stop - start) / max(points - 1, 1)
        direction = 1 if stop >= start else -1
        if direction > 0 and step < 0:
            step = abs(step)
        elif direction < 0 and step > 0:
            step = -abs(step)
        values = [start + i * step for i in range(points - 1)]
        values.append(stop)
        return values

    def _measurement_error(self, error: Exception) -> None:
        self._stop_event.set()
        self.start_btn.state(["!disabled"])
        self.stop_btn.state(["disabled"])
        self.summary_var.set("测量失败")
        messagebox.showerror("测量失败", f"测量过程中出现错误: {error}")
        self._log(f"测量失败: {error}")
        self._stop_event.clear()
        self._current_setpoints = []

    # ------------------------------------------------------------------
    def _append_point(self, point: MeasurementPoint) -> None:
        formatted_time = point.timestamp.strftime("%H:%M:%S")
        resistance = "∞" if math.isinf(point.resistance) else f"{point.resistance:.2f}"
        values = (
            point.index,
            formatted_time,
            f"{point.voltage:.3f}",
            f"{point.current:.6f}",
            resistance,
            f"{point.power:.6f}",
        )
        self.result_tree.insert("", tk.END, values=values)
        self._refresh_graph()
        self._update_summary()
        if self.auto_save_var.get():
            self._auto_save_snapshot()

    def _refresh_graph(self) -> None:
        self.graph.set_data(
            self.measurements,
            x_axis=self.x_axis_var.get(),
            y_axis=self.y_axis_var.get(),
        )

    def _update_summary(self) -> None:
        if not self.measurements:
            self.summary_var.set("准备就绪")
            return
        voltages = [point.voltage for point in self.measurements]
        currents = [point.current for point in self.measurements]
        avg_v = statistics.fmean(voltages)
        avg_i = statistics.fmean(currents)
        self.summary_var.set(f"已采集 {len(self.measurements)} 点 | 平均电压 {avg_v:.3f} V | 平均电流 {avg_i:.6f} A")

    def _measurement_finished(self) -> None:
        self.start_btn.state(["!disabled"])
        self.stop_btn.state(["disabled"])
        if self._stop_event.is_set():
            self.summary_var.set("测量已停止")
            self._log("测量被用户中断")
        else:
            self.summary_var.set("测量完成")
            self._log("测量完成")
        self._stop_event.clear()
        self._current_setpoints = []

    def stop_measurement(self) -> None:
        if self._measurement_thread and self._measurement_thread.is_alive():
            self._stop_event.set()
            try:
                self.instrument.abort_measurement()
            except Exception:
                pass
        self.start_btn.state(["!disabled"])
        self.stop_btn.state(["disabled"])
        self.summary_var.set("测量已停止")
        self._log("测量停止命令已执行")

    # ------------------------------------------------------------------
    def clear_results(self) -> None:
        if not self.measurements:
            return
        self.measurements.clear()
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        self._refresh_graph()
        self.summary_var.set("数据已清空")
        self._log("清空测量数据")

    def select_directory(self) -> None:
        directory = filedialog.askdirectory()
        if directory:
            self.file_path_var.set(directory)
            self._log(f"选择保存目录: {directory}")

    def _auto_save_snapshot(self) -> None:
        directory = self.file_path_var.get()
        if not directory:
            return
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = folder / f"auto_save_{timestamp}.csv"
        self._write_csv(filename)

    def save_results(self) -> None:
        if not self.measurements:
            messagebox.showinfo("提示", "当前没有可保存的数据")
            return
        filetypes = [("CSV 文件", "*.csv")]
        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=filetypes)
        if not filename:
            return
        self._write_csv(Path(filename))
        messagebox.showinfo("提示", "数据保存成功")
        self._log(f"保存数据到 {filename}")

    def _write_csv(self, filename: Path, points: Optional[Sequence[MeasurementPoint]] = None) -> None:
        data = list(points if points is not None else self.measurements)
        with filename.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["index", "timestamp", "voltage", "current", "resistance", "power"])
            for point in data:
                writer.writerow(
                    [
                        point.index,
                        point.timestamp.isoformat(),
                        f"{point.voltage:.6f}",
                        f"{point.current:.6f}",
                        "inf" if math.isinf(point.resistance) else f"{point.resistance:.6f}",
                        f"{point.power:.6f}",
                    ]
                )

    def export_configuration(self) -> None:
        settings = self.instrument.settings
        payload = {
            "timestamp": _dt.datetime.now().isoformat(),
            "measurement": self.measure_name_var.get(),
            "sample": self.sample_name_var.get(),
            "operator": self.operator_var.get(),
            "settings": settings.__dict__,
        }
        filetypes = [("JSON 文件", "*.json")]
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=filetypes)
        if not filename:
            return
        with open(filename, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        messagebox.showinfo("提示", "配置导出完成")
        self._log(f"导出配置到 {filename}")

    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        timestamp = _dt.datetime.now().strftime("%H:%M:%S")
        if not hasattr(self, "log_text"):
            return
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self._trim_log()
        self.log_text.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    def on_exit(self) -> None:
        if self.confirm_exit_var.get():
            if not messagebox.askokcancel("退出", "确定要退出程序吗?"):
                return
        self._stop_event.set()
        self._stop_all_time_tests()
        if self._script_thread and self._script_thread.is_alive():
            self._script_stop_event.set()
            self._script_thread.join(timeout=1.0)
        if self._measurement_thread and self._measurement_thread.is_alive():
            self._measurement_thread.join(timeout=1.0)
        try:
            if self.instrument.is_connected:
                self.instrument.disconnect()
        except Exception:
            pass
        self._save_preferences()
        self.destroy()


def main() -> None:
    app = MeasurementApp()
    app.mainloop()


if __name__ == "__main__":  # pragma: no cover
    main()
