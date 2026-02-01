
import sys
import inspect
from pathlib import Path
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.simulation_final2 import (
    AUDIO_PRESET_OPTIONS,
    make_audio_df,
    validate_audio_df,
    audio_df_to_totals,
    make_threshold_df,
    validate_threshold_df,
    build_threshold_series,
    run_period_monte_carlo_routing,
    # piecewise arrivals helpers
    make_piecewise_profile_df,
    validate_piecewise_profile_df,
    build_lambda_by_minute_piecewise,
)


st.set_page_config(page_title="Museum DSS - Crowding Analysis Dashboard", layout="wide")
st.title("Museum DSS - Crowding Analysis Dashboard")


if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = []

if "last_results" not in st.session_state:
    st.session_state["last_results"] = None

if "profile_key" not in st.session_state:
    st.session_state["profile_key"] = None


if "p_audio_compliant" not in st.session_state:
    st.session_state["p_audio_compliant"] = 0.97



def call_run_period(**kwargs):
    sig = inspect.signature(run_period_monte_carlo_routing)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return run_period_monte_carlo_routing(**filtered)


# Sidebar

st.sidebar.header("Planning settings")

n_rooms = 9
st.sidebar.metric("Rooms", n_rooms)

# Period
st.sidebar.divider()
st.sidebar.subheader("Period")
today = date.today()
d0 = st.sidebar.date_input("Start date", value=today - timedelta(days=6), key="d0")
d1 = st.sidebar.date_input("End date", value=today, key="d1")

if d1 < d0:
    st.sidebar.error("End date must be >= Start date.")
    st.stop()

period_days = (d1 - d0).days + 1
st.sidebar.metric("Days in period", period_days)

# Opening hours
st.sidebar.divider()
st.sidebar.subheader("Opening hours")
open_default = datetime.strptime("09:00", "%H:%M").time()
close_default = datetime.strptime("19:00", "%H:%M").time()

open_time = st.sidebar.time_input("Opening time", value=open_default, key="open_time")
close_time = st.sidebar.time_input("Closing time", value=close_default, key="close_time")

open_minute = int(open_time.hour * 60 + open_time.minute)
close_minute = int(close_time.hour * 60 + close_time.minute)

if close_minute <= open_minute:
    st.sidebar.error("Closing time must be after opening time.")
    st.stop()

daily_minutes = close_minute - open_minute

# Stay time model
st.sidebar.divider()
st.sidebar.subheader("Stay time model (Weibull)")

weibull_k = st.sidebar.number_input(
    "Shape k",
    min_value=0.8,
    max_value=5.0,
    value=float(st.session_state.get("weibull_k", 1.60)),
    step=0.10,
    format="%.2f",
    key="weibull_k_input",
)
st.session_state["weibull_k"] = float(weibull_k)

base_min_stay = st.sidebar.number_input(
    "Base stay per room (min)",
    min_value=0.0,
    max_value=10.0,
    value=float(st.session_state.get("base_min_stay", 1.00)),
    step=0.25,
    format="%.2f",
    key="base_min_stay_input",
)
st.session_state["base_min_stay"] = float(base_min_stay)

dwell_mult = st.sidebar.number_input(
    "Audio multiplier",
    min_value=0.0,
    max_value=5.0,
    value=float(st.session_state.get("dwell_mult", 1.30)),
    step=0.10,
    format="%.2f",
    key="dwell_mult_input",
)
st.session_state["dwell_mult"] = float(dwell_mult)

# Arrivals
st.sidebar.divider()
st.sidebar.subheader("Arrivals")

base_arrival_rate = st.sidebar.number_input(
    "Base arrival rate (visitors/min)",
    min_value=0.0,
    max_value=20.0,
    value=float(st.session_state.get("base_arrival_rate", 2.00)),
    step=0.10,
    format="%.2f",
    help="Baseline average arrival rate. Piecewise multipliers shape demand within the day.",
    key="base_arrival_rate_input",
)
st.session_state["base_arrival_rate"] = float(base_arrival_rate)

normalize_profile = st.sidebar.checkbox(
    "Normalize profile (keep average multiplier = 1)",
    value=bool(st.session_state.get("normalize_profile", True)),
    help="If enabled, multipliers rescaled so average is 1 during open hours.",
    key="normalize_profile_chk",
)
st.session_state["normalize_profile"] = bool(normalize_profile)

# Reset piecewise profiles if opening hours change
profile_key = (open_minute, close_minute)
if st.session_state.get("profile_key") != profile_key:
    st.session_state["weekday_piecewise_df"] = make_piecewise_profile_df(open_minute, close_minute, kind="weekday")
    st.session_state["weekend_piecewise_df"] = make_piecewise_profile_df(open_minute, close_minute, kind="weekend")
    st.session_state["profile_key"] = profile_key

weekday_piecewise_df = st.session_state["weekday_piecewise_df"]
weekend_piecewise_df = st.session_state["weekend_piecewise_df"]

st.sidebar.markdown("Weekday vs Weekend piecewise intervals (hours)")

with st.sidebar.expander("Edit weekday arrivals (Piecewise)", expanded=False):
    edited_wd_pw_h = st.data_editor(
        weekday_piecewise_df,
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Start hour": st.column_config.NumberColumn("Start hour", min_value=0.0, max_value=24.0, step=0.25),
            "End hour": st.column_config.NumberColumn("End hour", min_value=0.0, max_value=24.0, step=0.25),
            "Multiplier": st.column_config.NumberColumn("Multiplier", min_value=0.0, max_value=5.0, step=0.1),
        },
        key="weekday_piecewise_editor_hours",
    )
    if st.button("Apply weekday piecewise", key="btn_apply_wd_pw"):
        st.session_state["weekday_piecewise_df"] = validate_piecewise_profile_df(
            edited_wd_pw_h, open_minute, close_minute
        )
        st.success("Weekday piecewise profile applied.")

with st.sidebar.expander("Edit weekend arrivals (Piecewise)", expanded=False):
    edited_we_pw_h = st.data_editor(
        weekend_piecewise_df,
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Start hour": st.column_config.NumberColumn("Start hour", min_value=0.0, max_value=24.0, step=0.25),
            "End hour": st.column_config.NumberColumn("End hour", min_value=0.0, max_value=24.0, step=0.25),
            "Multiplier": st.column_config.NumberColumn("Multiplier", min_value=0.0, max_value=5.0, step=0.1),
        },
        key="weekend_piecewise_editor_hours",
    )
    if st.button("Apply weekend piecewise", key="btn_apply_we_pw"):
        st.session_state["weekend_piecewise_df"] = validate_piecewise_profile_df(
            edited_we_pw_h, open_minute, close_minute
        )
        st.success("Weekend piecewise profile applied.")


st.sidebar.divider()
st.sidebar.subheader("Visitor movement behavior")

p_audio_compliant = st.sidebar.slider(
    "Audio-guide users (%)",
    0, 100,
    int(round(st.session_state["p_audio_compliant"] * 100)),
    1,
    help="Share of visitors who follow the audio-guide suggested sequence (mostly Room 1→…→9). The rest explore more freely.",
    key="p_audio_compliant_slider_pct",
) / 100.0
st.session_state["p_audio_compliant"] = float(p_audio_compliant)


# Audio configuration
st.sidebar.divider()
st.sidebar.subheader("Audio guide durations")

if "audio_df" not in st.session_state:
    st.session_state["audio_df"] = make_audio_df(n_rooms, default_minutes=6.0)

audio_column_cfg = {
    "Audio preset": st.column_config.SelectboxColumn(
        "Audio preset",
        options=AUDIO_PRESET_OPTIONS,
        help="Pick 5/6/7 minutes or choose Custom",
    ),
    "Custom minutes": st.column_config.NumberColumn(
        "Custom minutes",
        min_value=0.5,
        max_value=30.0,
        step=0.25,
        help="Used only if preset = Custom",
    ),
}

with st.sidebar.form("audio_form_v2"):
    edited_audio = st.data_editor(
        st.session_state["audio_df"],
        hide_index=True,
        use_container_width=True,
        column_config=audio_column_cfg,
        disabled=["Room"],
        num_rows="fixed",
        key="audio_editor_v2",
    )
    apply_audio = st.form_submit_button("Apply audio changes")

if apply_audio:
    st.session_state["audio_df"] = validate_audio_df(edited_audio, n_rooms)
    st.sidebar.success("Audio settings applied.")

audio_df = st.session_state["audio_df"]
audio_totals = audio_df_to_totals(audio_df)

# Crowding thresholds
st.sidebar.divider()
st.sidebar.subheader("Crowding thresholds")

threshold_mode = st.sidebar.radio(
    "Threshold mode",
    ["Simple", "Advanced (per-room thresholds)"],
    index=0,
    key="thr_mode",
)

DEFAULT_SIMPLE_THR = 40
advanced_thr_df = None

if threshold_mode == "Simple":
    rooms_thr = st.sidebar.number_input(
        "Rooms threshold (visitors)",
        min_value=1,
        max_value=500,
        value=int(st.session_state.get("rooms_threshold_simple", DEFAULT_SIMPLE_THR)),
        step=1,
        key="simple_thr",
    )
    st.session_state["rooms_threshold_simple"] = int(rooms_thr)
    global_thr = int(rooms_thr)

else:
    if "thr_df" not in st.session_state:
        st.session_state["thr_df"] = make_threshold_df(n_rooms, DEFAULT_SIMPLE_THR)

    with st.sidebar.form("thr_form_v2"):
        edited_thr = st.data_editor(
            st.session_state["thr_df"],
            hide_index=True,
            use_container_width=True,
            disabled=["Room"],
            num_rows="fixed",
            column_config={
                "Threshold": st.column_config.NumberColumn("Threshold", min_value=1, max_value=500, step=1)
            },
            key="thr_editor_v2",
        )
        apply_thr = st.form_submit_button("Apply thresholds")

    if apply_thr:
        st.session_state["thr_df"] = validate_threshold_df(edited_thr, n_rooms, DEFAULT_SIMPLE_THR)
        st.sidebar.success("Thresholds applied.")
    else:
        st.session_state["thr_df"] = validate_threshold_df(edited_thr, n_rooms, DEFAULT_SIMPLE_THR)

    advanced_thr_df = st.session_state["thr_df"]
    global_thr = DEFAULT_SIMPLE_THR

thr_series = build_threshold_series(
    mode=threshold_mode,
    n_rooms=n_rooms,
    global_thr=int(global_thr),
    advanced_thr_df=advanced_thr_df,
)


# Plot helpers

def plot_heatmap(df: pd.DataFrame, open_minute: int, close_minute: int, bin_minutes: int, title: str, cbar_label: str):
    if df is None or df.empty:
        st.info("Heatmap not available.")
        return

    room_cols = [c for c in df.columns if c.startswith("Room ")]
    if not room_cols:
        st.info("Heatmap not available.")
        return

    df = df[(df["minute"] >= open_minute) & (df["minute"] <= close_minute)].copy()
    if df.empty:
        st.info("Heatmap not available.")
        return

    df["bin"] = ((df["minute"] - open_minute) // bin_minutes).astype(int)
    n_bins = int(((close_minute - open_minute) // bin_minutes) + 1)

    binned = df.groupby("bin")[room_cols].mean()
    binned = binned.reindex(range(n_bins)).fillna(0.0)

    mat = binned.to_numpy(dtype=float).T

    red_cmap = LinearSegmentedColormap.from_list(
        "reds_custom",
        ["#ffe5e5", "#ffb3b3", "#ff6666", "#cc0000"],
    )

    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", origin="upper", cmap=red_cmap, vmin=0)

    ax.set_title(title, loc="left", pad=10)
    ax.set_xlabel("Time of day")
    ax.set_ylabel("Room")

    ax.set_yticks(np.arange(len(room_cols)))
    ax.set_yticklabels(room_cols)

    bins_per_hour = max(1, int(60 // bin_minutes))
    xticks = np.arange(0, n_bins, bins_per_hour)
    ax.set_xticks(xticks)
    minutes_ticks = open_minute + xticks * bin_minutes
    ax.set_xticklabels([f"{m // 60:02d}:{m % 60:02d}" for m in minutes_ticks])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(fig, clear_figure=True)

def plot_arrivals_profile_piecewise(base_arrival_rate: float, normalize_profile: bool, open_minute: int, close_minute: int):
    wd_df = st.session_state["weekday_piecewise_df"]
    we_df = st.session_state["weekend_piecewise_df"]

    lam_wd = build_lambda_by_minute_piecewise(
        base_arrival_rate, open_minute, close_minute, wd_df, normalize=normalize_profile
    )
    lam_we = build_lambda_by_minute_piecewise(
        base_arrival_rate, open_minute, close_minute, we_df, normalize=normalize_profile
    )

    minutes = np.arange(open_minute, close_minute)
    hours = (minutes // 60).astype(int)
    df = pd.DataFrame({"Hour": hours, "Weekday": lam_wd, "Weekend": lam_we}).groupby("Hour").mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    ax.plot(df["Hour"], df["Weekday"], linewidth=2.0, label="Weekday")
    ax.plot(df["Hour"], df["Weekend"], linewidth=2.0, label="Weekend")
    ax.set_title("Arrival rate profile (Piecewise) — avg visitors/min by hour", loc="left", pad=10)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Visitors / minute")
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    st.pyplot(fig, clear_figure=True)

def plot_total_occupancy_bands(bands_total_df: pd.DataFrame, example_day: date):
    if bands_total_df is None or bands_total_df.empty:
        st.info("Total occupancy bands not available.")
        return

    minutes = bands_total_df["minute"].astype(int).to_numpy()
    x_dt = [datetime.combine(example_day, datetime.min.time()) + timedelta(minutes=int(m)) for m in minutes]

    y50 = bands_total_df["Total P50"].to_numpy(dtype=float)
    y90 = bands_total_df["Total P90"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    ax.plot(x_dt, y50, linewidth=2.0, label="Typical day (median)")
    ax.plot(x_dt, y90, linewidth=2.0, label="Busy day (90th percentile)")
    ax.fill_between(x_dt, y50, y90, alpha=0.18)

    ax.set_title("Total museum occupancy bands", loc="left", pad=10)
    ax.set_xlabel("Time (HH:MM)")
    ax.set_ylabel("Visitors")
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=60))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

    st.pyplot(fig, clear_figure=True)

def plot_scenario_tradeoff(scen_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.scatter(scen_df["Total overload min (mean)"], scen_df["Max occ (95th percentile)"], alpha=0.85)

    for _, row in scen_df.iterrows():
        ax.annotate(
            str(row["Name"]),
            (float(row["Total overload min (mean)"]), float(row["Max occ (95th percentile)"])),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    ax.set_title("Scenario trade-off (lower-left is better)", loc="left", pad=10)
    ax.set_xlabel("Total overload minutes (mean)")
    ax.set_ylabel("Max occupancy (95th percentile)")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig, clear_figure=True)

def plot_scenario_audio_impact(scen_audio_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.scatter(scen_audio_df["Avg audio (min)"], scen_audio_df["Total overload min (mean)"], alpha=0.85)

    for _, row in scen_audio_df.iterrows():
        ax.annotate(
            str(row["Name"]),
            (float(row["Avg audio (min)"]), float(row["Total overload min (mean)"])),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    ax.set_title("Scenario impact — Audio duration vs Overload", loc="left", pad=10)
    ax.set_xlabel("Avg audio duration across rooms (min)")
    ax.set_ylabel("Total overload minutes (mean)")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig, clear_figure=True)


# Recommendations helper

def propose_audio_adjustment_app(
    audio_df: pd.DataFrame,
    period_kpis: pd.DataFrame,
    top_k_rooms: int = 3,
    step_minutes: float = 1.0,
    min_audio: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if period_kpis is None or period_kpis.empty:
        return audio_df.copy(), pd.DataFrame()

    new_df = audio_df.copy()
    top_rooms = period_kpis.head(int(top_k_rooms))["Room"].tolist()

    changes = []
    for room in top_rooms:
        idx = int(room.split(" ")[1]) - 1
        preset = str(new_df.loc[idx, "Audio preset"])
        custom = float(new_df.loc[idx, "Custom minutes"])

        if preset in ("5", "6", "7"):
            cur = float(preset)
        else:
            cur = float(custom)

        new_val = max(float(min_audio), cur - float(step_minutes))
        if new_val >= cur - 1e-9:
            continue

        if abs(new_val - 5.0) < 1e-9:
            new_df.loc[idx, "Audio preset"] = "5"
            new_df.loc[idx, "Custom minutes"] = 5.0
            to_str = "5"
        elif abs(new_val - 6.0) < 1e-9:
            new_df.loc[idx, "Audio preset"] = "6"
            new_df.loc[idx, "Custom minutes"] = 6.0
            to_str = "6"
        elif abs(new_val - 7.0) < 1e-9:
            new_df.loc[idx, "Audio preset"] = "7"
            new_df.loc[idx, "Custom minutes"] = 7.0
            to_str = "7"
        else:
            new_df.loc[idx, "Audio preset"] = "Custom"
            new_df.loc[idx, "Custom minutes"] = float(new_val)
            to_str = f"{new_val:.2f}"

        from_str = preset if preset in ("5", "6", "7") else f"{cur:.2f}"
        changes.append({"Room": room, "From": from_str, "To": to_str, "Reason": "High expected overload"})

    return new_df, pd.DataFrame(changes)




st.markdown("## Run analysis")

with st.form("run_planning_form"):
    colA, colB = st.columns([1.2, 1.2])

    with colA:
        n_runs = st.number_input(
            "Simulations per day (Monte Carlo runs)",
            min_value=0,
            max_value=300,
            value=int(st.session_state.get("n_runs", 10)),
            step=1,
            key="n_runs_input",
        )

    with colB:
        st.write("")
        st.write("")
        run_analysis = st.form_submit_button("Run analysis")

if run_analysis:
    st.session_state["n_runs"] = int(n_runs)

    if int(n_runs) == 0:
        st.warning("n_runs = 0 → No simulation will run. Please set at least 1.")
    else:
        with st.spinner("Running analysis..."):
            period_kpis, overall, example_occ_df, example_day, heatmaps, bands_rooms_df, bands_total_df, diagnostics_df = (
                call_run_period(
                    d0=d0,
                    d1=d1,
                    n_runs_per_day=int(n_runs),
                    base_arrival_rate=float(base_arrival_rate),
                    open_minute=open_minute,
                    close_minute=close_minute,
                    audio_totals=audio_totals,
                    weibull_k=float(weibull_k),
                    base_min_stay=float(base_min_stay),
                    dwell_mult=float(dwell_mult),
                    thr_series=thr_series,
                    default_thr=int(global_thr),
                    normalize_profile=bool(normalize_profile),
                    max_store_runs_for_bands=1000,
                    weekday_piecewise_df=st.session_state["weekday_piecewise_df"],
                    weekend_piecewise_df=st.session_state["weekend_piecewise_df"],
                    p_audio_compliant=float(st.session_state["p_audio_compliant"]),
                )
            )

            avg_audio = float(np.mean(audio_totals)) if len(audio_totals) else 0.0

            st.session_state["last_results"] = {
                "d0": d0,
                "d1": d1,
                "period_days": period_days,
                "period_kpis": period_kpis,
                "overall": overall,
                "example_occ_df": example_occ_df,
                "example_day": example_day,
                "heatmaps": heatmaps,
                "bands_rooms_df": bands_rooms_df,
                "bands_total_df": bands_total_df,
                "diagnostics_df": diagnostics_df,
                "settings": {
                    "arrival_rate_base": float(base_arrival_rate),
                    "weibull_k": float(weibull_k),
                    "base_min_stay": float(base_min_stay),
                    "dwell_mult": float(dwell_mult),
                    "threshold_mode": threshold_mode,
                    "global_thr": int(global_thr),
                    "open_time": open_time.strftime("%H:%M"),
                    "close_time": close_time.strftime("%H:%M"),
                    "n_runs": int(n_runs),
                    "normalize_profile": bool(normalize_profile),
                },
                "audio_df": audio_df.copy(),
                "audio_totals": audio_totals.copy(),
                "avg_audio_min": avg_audio,
                "thr_series": thr_series.copy(),
            }

if st.session_state["last_results"] is None:
    st.info("Set inputs and run the analysis to see results.")
    st.stop()

res = st.session_state["last_results"]
period_kpis = res["period_kpis"]
overall = res["overall"]
example_occ_df = res["example_occ_df"]
example_day = res["example_day"]
heatmaps = res.get("heatmaps", {})
bands_rooms_df = res.get("bands_rooms_df", pd.DataFrame())
bands_total_df = res.get("bands_total_df", pd.DataFrame())


# Tabs

tab_overview, tab_rooms, tab_scenarios, tab_reco, tab_export = st.tabs(
    ["Overview", "Rooms", "Scenarios", "Recommendations", "Export"]
)

def display_kpi_table(df: pd.DataFrame):
    if df is None or df.empty:
        st.warning("No KPI produced (check parameters).")
        return
    view = df.copy()
    view.columns = [c.replace("P95", "95th percentile") for c in view.columns]
    st.dataframe(view, use_container_width=True, hide_index=True)


# Overview

with tab_overview:
    st.subheader(f"Period overview — {res['d0']} → {res['d1']} ({res['period_days']} days)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Worst room", overall["worst_room"])
    c2.metric("Total overload minutes (mean)", overall["total_overload_mean"])
    c3.metric("Peak occupancy (95th percentile)", overall["max_occ_p95"])
    c4.metric("Avg audio duration", f"{res['avg_audio_min']:.2f} min")

    st.divider()
    st.markdown("### Arrivals (Piecewise)")
    plot_arrivals_profile_piecewise(
        base_arrival_rate=float(res["settings"]["arrival_rate_base"]),
        normalize_profile=bool(res["settings"]["normalize_profile"]),
        open_minute=open_minute,
        close_minute=close_minute,
    )

    st.divider()
    st.markdown("### Total museum occupancy")
    plot_total_occupancy_bands(bands_total_df, example_day=example_day)

    st.divider()
    st.markdown("### Room KPI ranking (period-level)")
    if period_kpis.empty:
        st.warning("No KPI produced (check parameters).")
    else:
        sort_map = {
            "Minutes above threshold (mean)": "Minutes above thr (mean)",
            "Max occupancy (95th percentile)": "Max occupancy (P95)",
            "P(exceed) per run": "P(exceed) per run",
        }
        sort_choice_label = st.selectbox(
            "Sort by",
            list(sort_map.keys()),
            index=0,
            key="kpi_sort_overview",
        )
        sort_col = sort_map[sort_choice_label]
        kpi_sorted = period_kpis.sort_values(by=sort_col, ascending=False)
        display_kpi_table(kpi_sorted)

    st.divider()
    st.markdown("### Overcrowding heatmaps")

    colH1, colH2, colH3 = st.columns([1.1, 1.2, 1.0])
    with colH1:
        heat_day = st.selectbox("Days", ["All days", "Weekdays", "Weekends"], index=0, key="heat_day")
    with colH2:
        heat_metric = st.selectbox(
            "Metric",
            ["Mean excess above threshold", "Probability of exceeding threshold"],
            index=0,
            key="heat_metric",
        )
    with colH3:
        bin_minutes = st.selectbox("Bin size (minutes)", [5, 10, 15, 20, 30], index=1, key="heat_bin")

    if heat_day == "All days":
        excess_df = heatmaps.get("excess_all")
        prob_df = heatmaps.get("prob_all")
    elif heat_day == "Weekdays":
        excess_df = heatmaps.get("excess_weekday")
        prob_df = heatmaps.get("prob_weekday")
    else:
        excess_df = heatmaps.get("excess_weekend")
        prob_df = heatmaps.get("prob_weekend")

    if heat_metric == "Mean excess above threshold":
        plot_heatmap(
            excess_df,
            open_minute=open_minute,
            close_minute=close_minute,
            bin_minutes=int(bin_minutes),
            title=f"Overcrowding heatmap (mean excess) — {heat_day}",
            cbar_label="Mean visitors above threshold",
        )
    else:
        plot_heatmap(
            prob_df,
            open_minute=open_minute,
            close_minute=close_minute,
            bin_minutes=int(bin_minutes),
            title=f"Overcrowding heatmap (P(exceed)) — {heat_day}",
            cbar_label="Probability of exceeding threshold",
        )


# Rooms

with tab_rooms:
    st.subheader("Room Analysis")

    if period_kpis.empty:
        st.info("No data available.")
        st.stop()

    room_list = period_kpis["Room"].tolist()
    room_pick = st.selectbox("Select room", room_list, index=0, key="room_pick")

    k = period_kpis[period_kpis["Room"] == room_pick].iloc[0]
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Threshold", int(k["Threshold"]))
    a2.metric("Minutes above (mean)", int(round(k["Minutes above thr (mean)"])))
    a3.metric("Minutes above (95th percentile)", int(round(k["Minutes above thr (P95)"])))
    a4.metric("P(exceed) per run", f"{k['P(exceed) per run']}%")

    st.divider()
    st.markdown("### Room occupancy (Typical / Busy day)")
    if bands_rooms_df is None or bands_rooms_df.empty:
        st.info("Room bands not available.")
    else:
        p50_col = f"{room_pick} P50"
        p90_col = f"{room_pick} P90"
        if p50_col in bands_rooms_df.columns and p90_col in bands_rooms_df.columns:
            thr_plot = float(res["thr_series"].get(room_pick, res["settings"]["global_thr"]))
            minutes = bands_rooms_df["minute"].astype(int).to_numpy()
            x_dt = [datetime.combine(example_day, datetime.min.time()) + timedelta(minutes=int(m)) for m in minutes]

            y50 = bands_rooms_df[p50_col].to_numpy(dtype=float)
            y90 = bands_rooms_df[p90_col].to_numpy(dtype=float)

            fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
            ax.plot(x_dt, y50, linewidth=2.0, label="Typical day (median)")
            ax.plot(x_dt, y90, linewidth=2.0, label="Busy day (90th percentile)")
            ax.fill_between(x_dt, y50, y90, alpha=0.18)
            ax.axhline(thr_plot, linestyle="--", linewidth=1.2, label=f"Threshold ({int(thr_plot)})")

            ax.set_title(f"{room_pick} occupancy bands", loc="left", pad=10)
            ax.set_ylabel("Visitors")
            ax.set_xlabel("Time (HH:MM)")
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=60))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.grid(True, axis="y", alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False, loc="upper left")
            st.pyplot(fig, clear_figure=True)

    st.divider()
    st.markdown("### Example single-run occupancy curve")
    if example_occ_df is None or example_occ_df.empty:
        st.info("No example occupancy available.")
    else:
        room_cols = [c for c in example_occ_df.columns if c.startswith("Room ")]
        room_plot = st.selectbox("Plot room (single run)", room_cols, index=0, key="room_plot_example")

        thr_plot = float(res["thr_series"].get(room_plot, res["settings"]["global_thr"]))
        df = example_occ_df.copy()
        df = df[(df["minute"] >= open_minute) & (df["minute"] <= close_minute)].copy()

        x_minutes = df["minute"].astype(int).to_numpy()
        x_dt = [datetime.combine(example_day, datetime.min.time()) + timedelta(minutes=int(m)) for m in x_minutes]
        y = df[room_plot].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
        ax.plot(x_dt, y, linewidth=2.2, label="Occupancy")
        ax.axhline(thr_plot, linestyle="--", linewidth=1.2, label=f"Threshold ({int(thr_plot)})")

        ax.set_title(f"{room_plot} — example run", loc="left", pad=10)
        ax.set_ylabel("Visitors")
        ax.set_xlabel("Time (HH:MM)")
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=60))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="upper left")
        st.pyplot(fig, clear_figure=True)


# Scenarios

with tab_scenarios:
    st.subheader("Scenario planner")

    st.markdown("### Current scenario settings")
    st.write(res["settings"])
    st.write({"Avg audio minutes": float(res["avg_audio_min"])})

    with st.expander("Save current scenario", expanded=False):
        default_name = f"Scenario {len(st.session_state['scenarios']) + 1}"
        name = st.text_input("Scenario name", value=default_name, key="scenario_name")

        if st.button("Save scenario", key="btn_save_scenario"):
            st.session_state["scenarios"].append(
                {
                    "name": name,
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "settings": res["settings"],
                    "audio_df": res["audio_df"].copy(),
                    "avg_audio_min": float(res["avg_audio_min"]),
                    "period_kpis": period_kpis.copy(),
                    "overall": overall.copy(),
                }
            )
            st.success(f"Saved scenario: {name}")

    st.divider()
    st.markdown("### Scenario trade-off")
    scen_rows = [{
        "Name": "Current",
        "Total overload min (mean)": float(overall["total_overload_mean"]),
        "Max occ (95th percentile)": float(overall["max_occ_p95"]),
    }]
    for s in st.session_state["scenarios"]:
        scen_rows.append({
            "Name": s["name"],
            "Total overload min (mean)": float(s["overall"]["total_overload_mean"]),
            "Max occ (95th percentile)": float(s["overall"]["max_occ_p95"]),
        })

    scen_df = pd.DataFrame(scen_rows)
    plot_scenario_tradeoff(scen_df)

    st.divider()
    st.markdown("### Scenario impact of audio durations")
    audio_rows = [{
        "Name": "Current",
        "Avg audio (min)": float(res["avg_audio_min"]),
        "Total overload min (mean)": float(overall["total_overload_mean"]),
    }]
    for s in st.session_state["scenarios"]:
        audio_rows.append({
            "Name": s["name"],
            "Avg audio (min)": float(s["avg_audio_min"]),
            "Total overload min (mean)": float(s["overall"]["total_overload_mean"]),
        })
    scen_audio_df = pd.DataFrame(audio_rows)
    plot_scenario_audio_impact(scen_audio_df)


# Recommendations

with tab_reco:
    st.subheader("Recommendations")

    if period_kpis.empty:
        st.info("No recommendation available (no KPI data).")
    else:
        top_k = st.slider("How many rooms to adjust?", 1, n_rooms, 3, 1, key="reco_topk")
        step_minutes = st.selectbox("Custom reduction step (minutes)", [0.5, 1.0, 1.5, 2.0], index=1, key="reco_step")

        proposed_df, changes_df = propose_audio_adjustment_app(
            audio_df=audio_df,
            period_kpis=period_kpis,
            top_k_rooms=int(top_k),
            step_minutes=float(step_minutes),
            min_audio=3.0,
        )

        if changes_df.empty:
            st.info("No changes suggested (no overload or already at minimum).")
        else:
            st.markdown("### Suggested changes")
            st.dataframe(changes_df, use_container_width=True, hide_index=True)

            st.markdown("### Proposed audio plan (preview)")
            st.dataframe(proposed_df, use_container_width=True, hide_index=True)

            colR1, colR2 = st.columns([1.0, 1.0])
            with colR1:
                if st.button("Apply suggested plan to editor", key="btn_apply_reco"):
                    st.session_state["audio_df"] = proposed_df
                    st.success("Applied suggested plan. Now re-run analysis.")

            with colR2:
                if st.button("Evaluate suggested plan (A/B test)", key="btn_eval_reco"):
                    with st.spinner("Running A/B evaluation..."):
                        cur_kpis, cur_overall, *_ = call_run_period(
                            d0=res["d0"],
                            d1=res["d1"],
                            n_runs_per_day=int(res["settings"]["n_runs"]),
                            base_arrival_rate=float(res["settings"]["arrival_rate_base"]),
                            open_minute=open_minute,
                            close_minute=close_minute,
                            audio_totals=audio_df_to_totals(audio_df),
                            weibull_k=float(res["settings"]["weibull_k"]),
                            base_min_stay=float(res["settings"]["base_min_stay"]),
                            dwell_mult=float(res["settings"]["dwell_mult"]),
                            thr_series=res["thr_series"],
                            default_thr=int(res["settings"]["global_thr"]),
                            normalize_profile=bool(res["settings"]["normalize_profile"]),
                            max_store_runs_for_bands=0,
                            weekday_piecewise_df=st.session_state["weekday_piecewise_df"],
                            weekend_piecewise_df=st.session_state["weekend_piecewise_df"],
                            p_audio_compliant=float(st.session_state["p_audio_compliant"]),
                        )

                        prop_kpis, prop_overall, *_ = call_run_period(
                            d0=res["d0"],
                            d1=res["d1"],
                            n_runs_per_day=int(res["settings"]["n_runs"]),
                            base_arrival_rate=float(res["settings"]["arrival_rate_base"]),
                            open_minute=open_minute,
                            close_minute=close_minute,
                            audio_totals=audio_df_to_totals(proposed_df),
                            weibull_k=float(res["settings"]["weibull_k"]),
                            base_min_stay=float(res["settings"]["base_min_stay"]),
                            dwell_mult=float(res["settings"]["dwell_mult"]),
                            thr_series=res["thr_series"],
                            default_thr=int(res["settings"]["global_thr"]),
                            normalize_profile=bool(res["settings"]["normalize_profile"]),
                            max_store_runs_for_bands=0,
                            weekday_piecewise_df=st.session_state["weekday_piecewise_df"],
                            weekend_piecewise_df=st.session_state["weekend_piecewise_df"],
                            p_audio_compliant=float(st.session_state["p_audio_compliant"]),
                        )

                    st.markdown("### A/B results (Current vs Proposed)")
                    delta_overload = float(prop_overall["total_overload_mean"]) - float(cur_overall["total_overload_mean"])
                    delta_95p = float(prop_overall["max_occ_p95"]) - float(cur_overall["max_occ_p95"])

                    cA, cB, cC = st.columns(3)
                    cA.metric("Δ overload minutes", f"{delta_overload:.0f}")
                    cB.metric("Δ max occupancy (95th percentile)", f"{delta_95p:.0f}")
                    cC.metric("Worst room (proposed)", str(prop_overall["worst_room"]))

                    if delta_overload < 0 and delta_95p <= 0:
                        st.success(" Suggested plan improves overload and does not increase peak risk.")
                    elif delta_overload < 0:
                        st.warning(" Overload improves, but peak increases slightly. Consider smaller changes.")
                    else:
                        st.error(" Suggested plan did not improve overload. Try tuning top_k or step.")


# Export

with tab_export:
    st.subheader("Export results")

    if period_kpis.empty:
        st.info("No export available.")
    else:
        csv = period_kpis.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download period KPI table (CSV)",
            data=csv,
            file_name=f"museum_planning_kpis_{res['d0']}_{res['d1']}.csv",
            mime="text/csv",
            key="dl_kpi",
        )

    st.divider()
    st.markdown("### Export audio configuration")
    audio_csv = audio_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download audio plan (CSV)",
        data=audio_csv,
        file_name=f"audio_plan_{res['d0']}_{res['d1']}.csv",
        mime="text/csv",
        key="dl_audio",
    )

    st.divider()
    st.markdown("### Export thresholds")
    thr_export = pd.DataFrame({"Room": thr_series.index.tolist(), "Threshold": thr_series.values})
    thr_csv = thr_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download thresholds (CSV)",
        data=thr_csv,
        file_name=f"thresholds_{res['d0']}_{res['d1']}.csv",
        mime="text/csv",
        key="dl_thr",
    )