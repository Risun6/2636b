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
from typing import Dict, Iterable, List, Optional, Tuple


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
    def start_measurement(self) -> None:
        if not self._is_connected:
            raise RuntimeError("Instrument not connected")
        self._start_time = _dt.datetime.now()

    def generate_point(self, index: int) -> MeasurementPoint:
        if self._start_time is None:
            self._start_time = _dt.datetime.now()

        elapsed = (index - 1) * self.settings.step
        angle = (elapsed / max(self.settings.stop_level - self.settings.start_level, 0.1)) * math.tau
        base_voltage = math.sin(angle) * 5
        base_current = math.cos(angle) * 0.01

        noise_v = self._seed.uniform(-0.05, 0.05)
        noise_i = self._seed.uniform(-0.0002, 0.0002)

        voltage = base_voltage + noise_v
        current = base_current + noise_i

        timestamp = self._start_time + _dt.timedelta(seconds=elapsed)
        return MeasurementPoint(index=index, timestamp=timestamp, voltage=voltage, current=current)


# ---------------------------------------------------------------------------
# Graph rendering helper
# ---------------------------------------------------------------------------

class SweepGraph(ttk.Frame):
    """Simple canvas based plot that mimics the layout in the screenshots."""

    def __init__(self, master: tk.Widget, *, width: int = 840, height: int = 320) -> None:
        super().__init__(master)
        self.width = width
        self.height = height
        self.canvas = tk.Canvas(self, width=width, height=height, background="#ffffff", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.margin = 40
        self._data: List[Tuple[float, float]] = []
        self._x_range = (0.0, 10.0)
        self._y_range = (-1.0, 1.0)
        self.canvas.bind("<Configure>", lambda event: self.redraw())
        self.redraw()

    # ------------------------------------------------------------------
    def set_data(self, points: Iterable[MeasurementPoint], *, x_axis: str = "Voltage", y_axis: str = "Current") -> None:
        processed = []
        for point in points:
            x_value = point.voltage if x_axis == "Voltage" else point.index
            y_value = point.current if y_axis == "Current" else getattr(point, y_axis.lower(), point.current)
            processed.append((x_value, y_value))
        self._data = processed
        if processed:
            xs, ys = zip(*processed)
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            pad_x = (max_x - min_x) * 0.1 or 1
            pad_y = (max_y - min_y) * 0.1 or 1
            self._x_range = (min_x - pad_x, max_x + pad_x)
            self._y_range = (min_y - pad_y, max_y + pad_y)
        self.redraw()

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

        # Background grid
        self.canvas.create_rectangle(plot_left, plot_top, plot_right, plot_bottom, outline="#8d8d8d")
        for i in range(10):
            y = plot_top + (plot_bottom - plot_top) * (i / 10)
            self.canvas.create_line(plot_left, y, plot_right, y, fill="#e0e0e0")
        for i in range(10):
            x = plot_left + (plot_right - plot_left) * (i / 10)
            self.canvas.create_line(x, plot_top, x, plot_bottom, fill="#e0e0e0")

        # Axis labels
        self.canvas.create_text(width / 2, height - margin / 3, text="电压 (V)")
        self.canvas.create_text(margin / 3, height / 2, text="电流 (A)", angle=90)

        if not self._data:
            self.canvas.create_text(width / 2, height / 2, text="尚未开始测量", fill="#888888")
            return

        xs, ys = zip(*self._data)
        min_x, max_x = self._x_range
        min_y, max_y = self._y_range
        span_x = max(max_x - min_x, 1e-9)
        span_y = max(max_y - min_y, 1e-9)

        coordinates = []
        for x_value, y_value in zip(xs, ys):
            x = plot_left + ((x_value - min_x) / span_x) * (plot_right - plot_left)
            y = plot_bottom - ((y_value - min_y) / span_y) * (plot_bottom - plot_top)
            coordinates.append((x, y))

        for i in range(1, len(coordinates)):
            x0, y0 = coordinates[i - 1]
            x1, y1 = coordinates[i]
            self.canvas.create_line(x0, y0, x1, y1, fill="#0066cc", width=2)

        for x, y in coordinates:
            self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#0066cc", outline="")


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

        self.instrument = InstrumentSimulator()
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

        connect_btn = ttk.Button(connection_frame, text="连接仪器", command=self.connect_instrument)
        connect_btn.pack(fill=tk.X, pady=4)

        disconnect_btn = ttk.Button(connection_frame, text="断开连接", command=self.disconnect_instrument)
        disconnect_btn.pack(fill=tk.X, pady=4)

        ttk.Label(connection_frame, text="仪器状态:").pack(anchor="w", pady=(6, 0))
        ttk.Label(connection_frame, textvariable=self.device_status_var, foreground="#3073b6", font=("Microsoft YaHei", 11, "bold")).pack(anchor="w")

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

        self.graph = SweepGraph(graph_frame)
        self.graph.pack(fill=tk.BOTH, expand=True)

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
    def connect_instrument(self) -> None:
        if self.instrument.is_connected:
            messagebox.showinfo("提示", "仪器已连接")
            return
        try:
            self.instrument.connect()
        except Exception as exc:  # pragma: no cover - defensive guard
            messagebox.showerror("错误", f"连接仪器失败: {exc}")
            return
        self.device_status_var.set("已连接")
        self.summary_var.set("仪器连接成功")
        self._log("仪器已连接")

    def disconnect_instrument(self) -> None:
        if not self.instrument.is_connected:
            messagebox.showinfo("提示", "仪器未连接")
            return
        self.instrument.disconnect()
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
        self.graph.set_data([])
        self.timestamp_value.config(text=_dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.start_btn.state(["disabled"])
        self.stop_btn.state(["!disabled"])
        self._log("开始测量")

        self._measurement_thread = threading.Thread(target=self._run_measurement, daemon=True)
        self._measurement_thread.start()

    def _run_measurement(self) -> None:
        settings = self.instrument.settings
        total_points = settings.sweep_points
        self.instrument.start_measurement()
        for index in range(1, int(total_points) + 1):
            if self._stop_event.is_set():
                break
            point = self.instrument.generate_point(index)
            self.measurements.append(point)
            self.after(0, self._append_point, point)
            time.sleep(max(settings.step * 0.2, 0.2))
        self.after(0, self._measurement_finished)

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
        self.graph.set_data(self.measurements)
        self._update_summary()
        if self.auto_save_var.get():
            self._auto_save_snapshot()

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

    def stop_measurement(self) -> None:
        if self._measurement_thread and self._measurement_thread.is_alive():
            self._stop_event.set()
        self.start_btn.state(["!disabled"])
        self.stop_btn.state(["disabled"])

    # ------------------------------------------------------------------
    def clear_results(self) -> None:
        if not self.measurements:
            return
        self.measurements.clear()
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        self.graph.set_data([])
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
