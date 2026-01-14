# museum-dss

Streamlit Decision Support System (DSS) prototype to study museum crowding and how flexible audio-guide segments affect dwell time and occupancy.

## Project structure
- `dash/` : Streamlit UI (`app.py`)
- `src/`  : simulation logic (backend)
- `data/` : optional datasets (if any)
- `tests/`: tests (later)

## Setup (local)
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
