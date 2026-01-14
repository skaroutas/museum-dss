import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import sys

# Ensure project root is on PYTHONPATH so "src" imports work reliably when running Streamlit
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.simulation import simulate_museum_week1


st.set_page_config(page_title="Museum DSS – Demo", layout="wide")
st.title("Museum DSS – Demo Dashboard (Week 1)")
st.caption("Simple simulation-based decision support prototype: audio segments → dwell time → occupancy.")


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Simulation Settings")

arrival_rate = st.sidebar.slider("Arrival rate (visitors/min)", 0.5, 5.0, 2.0, 0.1)
simulation_time = st.sidebar.slider("Simulation time (minutes)", 30, 300, 120, 10)
n_rooms = st.sidebar.slider("Number of rooms", 3, 12, 6, 1)

st.sidebar.divider()
st.sidebar.subheader("Weibull stay time (per room)")
weibull_k = st.sidebar.slider("Shape k", 0.8, 5.0, 1.6, 0.1)
base_min_stay = st.sidebar.slider("Base stay per room (min)", 0.0, 5.0, 1.0, 0.5)
dwell_mult = st.sidebar.slider("Audio multiplier (E[stay] = base + mult×audio)", 0.5, 3.0, 1.3, 0.1)

st.sidebar.divider()
st.sidebar.subheader("Manager: choose audio segments (per room)")

st.sidebar.caption("Edit durations (minutes) for up to 5 segments per room. Use 0 for unused segments.")

# Create / keep an editable table in session state
if "segments_df" not in st.session_state or st.session_state["segments_df"].shape[0] != n_rooms:
    st.session_state["segments_df"] = pd.DataFrame({
        "Room": [f"Room {i+1}" for i in range(n_rooms)],
        "Seg 1": [2] * n_rooms,
        "Seg 2": [2] * n_rooms,
        "Seg 3": [0] * n_rooms,
        "Seg 4": [0] * n_rooms,
        "Seg 5": [0] * n_rooms,
    })

segments_df = st.sidebar.data_editor(
    st.session_state["segments_df"],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Seg 1": st.column_config.NumberColumn(min_value=0, max_value=5, step=1),
        "Seg 2": st.column_config.NumberColumn(min_value=0, max_value=5, step=1),
        "Seg 3": st.column_config.NumberColumn(min_value=0, max_value=5, step=1),
        "Seg 4": st.column_config.NumberColumn(min_value=0, max_value=5, step=1),
        "Seg 5": st.column_config.NumberColumn(min_value=0, max_value=5, step=1),
    }
)
st.session_state["segments_df"] = segments_df

audio_totals = segments_df[["Seg 1", "Seg 2", "Seg 3", "Seg 4", "Seg 5"]].sum(axis=1).to_numpy()

st.sidebar.divider()
overcrowd_threshold = st.sidebar.slider("Overcrowding threshold (visitors)", 1, 50, 10, 1)

run = st.sidebar.button("Run simulation", type="primary")


# -----------------------------
# Run / Compute
# -----------------------------
if run:
    visits_df, occ_df, mean_stay_per_room = simulate_museum_week1(
        arrival_rate=arrival_rate,
        simulation_time=simulation_time,
        n_rooms=n_rooms,
        audio_minutes_per_room=audio_totals,
        weibull_shape_k=weibull_k,
        base_min_stay=base_min_stay,
        dwell_multiplier=dwell_mult,
        seed=42
    )

    # KPIs
    room_cols = [c for c in occ_df.columns if c.startswith("Room ")]
    occ_only = occ_df[room_cols]
    max_occ = int(occ_only.to_numpy().max()) if not occ_only.empty else 0
    peak_room_idx = int(np.argmax(occ_only.to_numpy().max(axis=0))) if not occ_only.empty else 0
    peak_room = room_cols[peak_room_idx] if room_cols else "N/A"
    minutes_above = int((occ_only.to_numpy() > overcrowd_threshold).sum()) if not occ_only.empty else 0

    # Layout
    c1, c2, c3 = st.columns(3)
    c1.metric("Max occupancy (any room)", max_occ)
    c2.metric("Peak room", peak_room)
    c3.metric("Total room-minutes above threshold", minutes_above)

    st.divider()

    left, right = st.columns([1, 1])

    # -----------------------------
    # Left: Room graph
    # -----------------------------
    with left:
        st.subheader("Museum layout (demo room graph)")

        # Simple ring graph for Week 1 (keeps it easy)
        G = nx.cycle_graph(n_rooms)
        mapping = {i: f"R{i+1}" for i in range(n_rooms)}
        G = nx.relabel_nodes(G, mapping)

        fig = plt.figure()
        pos = nx.circular_layout(G)
        nx.draw(G, pos, with_labels=True)
        st.pyplot(fig, clear_figure=True)

        st.caption("Week 1: fixed demo layout (ring). Later: custom layouts from the manager.")

    # -----------------------------
    # Right: Occupancy plots
    # -----------------------------
    with right:
        st.subheader("Occupancy over time")

        chosen_room = st.selectbox("Select room to view", room_cols, index=0)
        fig = plt.figure()
        plt.plot(occ_df["minute"], occ_df[chosen_room])
        plt.axhline(overcrowd_threshold, linestyle="--")
        plt.xlabel("Minute")
        plt.ylabel("Occupancy")
        st.pyplot(fig, clear_figure=True)

        st.caption("Dashed line = overcrowding threshold.")

    st.divider()

    # Heatmap
    st.subheader("Rooms × time heatmap (occupancy)")
    heat = occ_only.to_numpy().T  # rooms x time
    fig = plt.figure()
    plt.imshow(heat, aspect="auto", interpolation="nearest")
    plt.yticks(range(n_rooms), [f"Room {i+1}" for i in range(n_rooms)])
    plt.xlabel("Minute index")
    plt.ylabel("Room")
    st.pyplot(fig, clear_figure=True)

    st.divider()

    # Show inputs + sample output tables
    st.subheader("Manager audio configuration (derived totals)")
    audio_summary = segments_df.copy()
    audio_summary["Total audio (min)"] = audio_summary[["Seg 1","Seg 2","Seg 3","Seg 4","Seg 5"]].sum(axis=1)
    audio_summary["E[stay] (min)"] = np.round(mean_stay_per_room, 2)
    st.dataframe(audio_summary, use_container_width=True)

    st.subheader("Sample simulated visits (first 20)")
    st.dataframe(visits_df.head(20), use_container_width=True)

else:
    st.info("Adjust the sidebar inputs and click **Run simulation**.")

