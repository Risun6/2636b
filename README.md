# 2636B Measurement Controller

This project delivers a desktop control program for Keithley 2636B source
meters that mirrors the workflow shown in the reference screenshots.  It now
supports both real hardware via PyVISA as well as a deterministic simulator for
offline use.  The application is built entirely on the Python standard library
UI toolkit (Tkinter) so it runs out of the box on any machine with Python
installed.

## Key Capabilities

* **Direct instrument connectivity** – scan connected VISA resources, display
  identity details, and establish a live session with a 2636B (or compatible)
  source meter.
* **Deterministic simulator fallback** – when no hardware is present the
  built-in instrument model provides repeatable synthetic measurements.
* **Robust sweep execution** – measurement sequencing respects sweep
  direction, trigger delays, compliance limits and user defined point counts.
* **Dynamic charting** – the plot now offers selectable axes (index, time,
  voltage, current, resistance, power) with auto scaling and nice tick labels.
* **Time-domain capture** – dedicated V/t 与 I/t 测试标签页支持自定义间隔、
  实时绘图、统计信息以及 CSV 导出，帮助快速观察漂移与稳定性。
* **Analysis utilities** – 阈值扫描和击穿测试提供一键阈值/斜率分析，
  输出详细提示并可导出 JSON 报告。
* **Comprehensive data handling** – live statistics, automatic log updates,
  CSV export, JSON configuration export and optional auto-save snapshots.
* **Manual control & automation** – 输出控制页面允许手动设定源表输出，
  脚本执行器支持 MODE/LEVEL/MEASURE/SWEEP/WAIT/LOG 指令串联复杂流程。
* **System preferences** – 主题切换、日志行数限制、启动行为等偏好现在
  可视化配置，并在用户目录中持久化保存。

## Requirements

* Python 3.9 or newer.
* Tkinter (bundled with most Python distributions).
* Optional: [PyVISA](https://pyvisa.readthedocs.io/) and a VISA backend (for
  example NI-VISA) to communicate with real instruments.

Install PyVISA if you plan to use hardware control:

```bash
pip install pyvisa
```

## Running and Connecting

Launch the UI with:

```bash
python main.py
```

1. Click **扫描设备** to enumerate all VISA resources.  Detected instruments and
   their identity strings appear in the dropdown list.
2. Select the desired resource (or keep **内置模拟器** to stay offline) and press
   **连接仪器**.  The status label will show the connected instrument.
3. Adjust sweep parameters, compliance limits and metadata in the right-hand
   panels.
4. Choose the preferred x/y axes above the chart, then press **开始** to launch
   the sweep.  Data points stream into the table and chart until the sequence
   completes or **停止** is pressed.
5. Use the sidebar buttons to clear data, export CSV/JSON files or enable
   automatic snapshot saving.
6. Explore the notebook tabs for V/t、I/t 采集、阈值/击穿分析、脚本执行以及
   系统设置等扩展功能。

If no VISA device is detected the application automatically falls back to the
simulator while still exposing the same workflow.

## Repository Layout

```
main.py    - GUI application with instrument drivers, plotting logic, time
            series tests, analysis tools and automation helpers
README.md  - Project overview, usage instructions and feature summary
```

The codebase remains intentionally compact and well commented to simplify
extension for additional measurement modes or instrument families.
