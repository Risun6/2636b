# 2636B Measurement Controller

A standalone desktop control app for Keithley 2636B source meters that mirrors
the reference workflow. It supports **real hardware via PyVISA** and a
**deterministic simulator** for offline use. The UI is built with the Python
standard library toolkit (Tkinter), so it runs out of the box.

## Key Capabilities

- **Direct instrument connectivity** – scan VISA resources, show identity
  strings, and open a live session with a 2636B (or compatible) sourcemeter.
- **Deterministic simulator fallback** – when no hardware is present, the
  built-in model provides repeatable synthetic measurements.
- **Robust sweep execution** – sequencing respects sweep direction, trigger
  delays, compliance limits, point counts, and user timing.
- **Dynamic charting** – selectable axes (index, time, voltage, current,
  resistance, power) with auto-scaling and readable ticks.
- **Results & data handling** – live statistics, auto log updates, CSV export,
  JSON configuration export, and optional auto-save snapshots.

## Requirements

- Python 3.9 or newer
- Tkinter (bundled with most Python distributions)
- Optional: [PyVISA](https://pyvisa.readthedocs.io/) and a VISA backend
  (e.g., NI-VISA) to communicate with real instruments

Install PyVISA if you plan to use hardware control:

```bash
pip install pyvisa
