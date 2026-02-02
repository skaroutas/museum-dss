# Museum DSS — Crowding Analysis Dashboard

A Streamlit-based **Decision Support System (DSS)** to analyze **museum crowding** and evaluate how **flexible audio-guide durations** and **visitor adherence** to the suggested sequence affect **dwell times**, **room occupancy**, and **overcrowding risk**.

The project includes:
- a **simulation back-end** (stochastic arrivals + routing + dwell times + exit behavior),
- an **interactive dashboard** for scenario configuration, Monte Carlo evaluation, visualization, and export.



## Repository Structure

- `src/`
  - `simulation_final2.py` — back-end engine: model assumptions, validation, sampling routines, simulation logic, KPI computation, Monte Carlo aggregation, heatmaps, occupancy bands.
- `dash/`
  - `app.py` — front-end: Streamlit app (inputs, plots, scenarios, recommendations, export).
- `Museum_DSS___Report.pdf` — final report (model + architecture + math + results).



## Requirements

- Python **3.9+** (recommended 3.10+)
- Main packages: `streamlit`, `numpy`, `pandas`, `matplotlib`

Install via `requirements.txt` (included in the repo).



## Setup (Local)

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt

streamlit run dash/app.py

