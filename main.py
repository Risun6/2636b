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
from dataclasses import dataclass
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

        self.canvas.create_rectangle(plot_left, plot_top, plot_right, plot_bottom, outline="#a0a0a0")

        self.canvas.create_text(width / 2, height - margin / 3, text=self._x_axis_label)
        self.canvas.create_text(margin / 3, height / 2, text=self._y_axis_label, angle=90)

        if not self._data:
            self.canvas.create_text(width / 2, height / 2, text="尚未开始测量", fill="#888888")
            return

        min_x, max_x = self._x_range
        min_y, max_y = self._y_range
        span_x = max(max_x - min_x, 1e-12)
        span_y = max(max_y - min_y, 1e-12)

        x_ticks = self._generate_ticks(min_x, max_x)
        y_ticks = self._generate_ticks(min_y, max_y)

        for value, label in x_ticks:
            position = plot_left + ((value - min_x) / span_x) * (plot_right - plot_left)
            self.canvas.create_line(position, plot_top, position, plot_bottom, fill="#f0f0f0")
            self.canvas.create_text(position, plot_bottom + 12, text=label, anchor="n", font=("Arial", 8))

        for value, label in y_ticks:
            position = plot_bottom - ((value - min_y) / span_y) * (plot_bottom - plot_top)
            self.canvas.create_line(plot_left, position, plot_right, position, fill="#f0f0f0")
            self.canvas.create_text(plot_left - 6, position, text=label, anchor="e", font=("Arial", 8))

        coordinates = []
        for x_value, y_value in self._data:
            x = plot_left + ((x_value - min_x) / span_x) * (plot_right - plot_left)
            y = plot_bottom - ((y_value - min_y) / span_y) * (plot_bottom - plot_top)
            coordinates.append((x, y))

        for i in range(1, len(coordinates)):
            x0, y0 = coordinates[i - 1]
            x1, y1 = coordinates[i]
            self.canvas.create_line(x0, y0, x1, y1, fill="#0066cc", width=2)

        for x, y in coordinates:
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#0066cc", outline="")


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

        self.instrument: InstrumentSimulator | VisaInstrument = InstrumentSimulator()
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

        self._build_layout()
        self._configure_styles()
        self.protocol("WM_DELETE_WINDOW", self.on_exit)

    # ------------------------------------------------------------------
    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", padding=6)
        style.configure("Primary.TButton", font=("Microsoft YaHei", 12, "bold"), foreground="#ffffff", background="#3073b6")
        style.map(
            "Primary.TButton",
            background=[("active", "#1e5590"), ("disabled", "#9bb7d7")],
        )
        style.configure("Danger.TButton", foreground="#ffffff", background="#d9534f")
        style.map("Danger.TButton", background=[("active", "#c9302c")])

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

        for tab_name in [
            "I/V测量",
            "V/t测试",
            "I/t测试",
            "阈值扫描",
            "击穿测试",
            "输出控制",
            "脚本执行",
            "系统设置",
        ][1:]:
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=tab_name)
            placeholder = ttk.Label(frame, text=f"{tab_name} 功能将在此显示", foreground="#777777")
            placeholder.pack(pady=40)

        self._build_iv_tab(iv_tab)

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
            self.instrument.prepare_measurement(self._current_setpoints)  # type: ignore[attr-defined]
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
                point = self.instrument.generate_point(index, level)  # type: ignore[attr-defined]
                self.measurements.append(point)
                self.after(0, self._append_point, point)
                time.sleep(dwell)
        except Exception as exc:  # pragma: no cover - depends on hardware
            self.after(0, lambda error=exc: self._measurement_error(error))
        else:
            self.after(0, self._measurement_finished)
        finally:
            try:
                self.instrument.finalize_measurement()  # type: ignore[attr-defined]
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
                self.instrument.abort_measurement()  # type: ignore[attr-defined]
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

    def _write_csv(self, filename: Path) -> None:
        with filename.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["index", "timestamp", "voltage", "current", "resistance", "power"])
            for point in self.measurements:
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
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    def on_exit(self) -> None:
        self._stop_event.set()
        if self._measurement_thread and self._measurement_thread.is_alive():
            self._measurement_thread.join(timeout=1.0)
        self.destroy()


def main() -> None:
    app = MeasurementApp()
    app.mainloop()


if __name__ == "__main__":  # pragma: no cover
    main()
