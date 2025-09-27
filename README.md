# 2636B Measurement Software Replica

This repository contains a standalone desktop application that reproduces the
workflow shown in the supplied screenshots for a Keithley 2636B control
program.  The implementation focuses on matching the layout and user
experience while providing a fully functional simulation back-end so the
software can be demonstrated without lab hardware.

## Features

* Multi-tab interface for I/V sweeps, time-domain tests and utility pages.
* Fully simulated instrument driver with deterministic synthetic data.
* Canvas-based chart replicating the grid and trace appearance from the
  original application.
* Measurement results table with live updates, averaging and auto-save.
* Parameter panels for source settings, sweep ranges, compliance limits and
  logging.
* Export utilities for saving captured data (CSV) and measurement
  configuration (JSON).

## Running the Application

The program only depends on the Python standard library.  Launch it directly:

```bash
python main.py
```

When the UI opens:

1. Click **连接仪器** to attach the simulated source meter.
2. Configure sweep parameters from the panels on the right.
3. Press **开始** to start collecting samples.  The graph and table update in
   real time until all sweep points are acquired or **停止** is pressed.
4. Save the generated dataset or export configuration files using the buttons
   in the left sidebar.

All measurement points are produced by a deterministic mathematical model, so
repeat runs using the same settings yield reproducible curves and statistics.

## Repository Layout

```
main.py    - GUI application entry point
README.md  - Project overview and instructions
```

The codebase was kept intentionally compact to emphasise clarity and ease of
modification.
