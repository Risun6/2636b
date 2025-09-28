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
* **Comprehensive data handling** – live statistics, automatic log updates,
  CSV export, JSON configuration export and optional auto-save snapshots.

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

If no VISA device is detected the application automatically falls back to the
simulator while still exposing the same workflow.

## Repository Layout

```
main.py    - GUI application with instrument drivers and plotting logic
README.md  - Project overview and usage instructions
```

The codebase remains intentionally compact and well commented to simplify
extension for additional measurement modes or instrument families.
