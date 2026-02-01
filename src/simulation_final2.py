
from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _weibull_scale_for_mean(mean: float, shape_k: float) -> float:
    """mean = scale * Gamma(1 + 1/k) => scale = mean / Gamma(1 + 1/k)"""
    mean = max(float(mean), 0.1)
    return mean / math.gamma(1.0 + 1.0 / shape_k)


def _compute_mean_stay(
    audio_minutes_per_room: np.ndarray,
    base_min_stay: float,
    dwell_multiplier: float,
) -> np.ndarray:
    mean_stay = float(base_min_stay) + float(dwell_multiplier) * np.asarray(audio_minutes_per_room, dtype=float)
    return np.clip(mean_stay, 0.5, None)


def _validate_audio(audio_minutes_per_room: Optional[np.ndarray], n_rooms: int) -> np.ndarray:
    """Ensure audio array exists and has length n_rooms."""
    if audio_minutes_per_room is None:
        audio_minutes_per_room = np.full(n_rooms, 6.0, dtype=float)
    audio_minutes_per_room = np.asarray(audio_minutes_per_room, dtype=float)

    if audio_minutes_per_room.shape[0] != n_rooms:
        raise ValueError(
            f"audio_minutes_per_room must have length {n_rooms}, "
            f"got {audio_minutes_per_room.shape[0]}"
        )
    return audio_minutes_per_room


def _occupancy_from_intervals(visits_df: pd.DataFrame, n_rooms: int, simulation_time: int) -> pd.DataFrame:
    """visits_df needs: room_index, entry_time, exit_time. Returns occupancy on minute grid t=0..T."""
    t = np.arange(0, simulation_time + 1)
    occ = np.zeros((n_rooms, len(t)), dtype=int)

    for r in range(n_rooms):
        rr = visits_df[visits_df["room_index"] == r][["entry_time", "exit_time"]].to_numpy()
        if rr.size == 0:
            continue
        a = rr[:, 0][:, None]
        e = rr[:, 1][:, None]
        occ[r, :] = np.sum((a <= t) & (t < e), axis=0)

    occ_df = pd.DataFrame(occ.T, columns=[f"Room {i+1}" for i in range(n_rooms)])
    occ_df.insert(0, "minute", t)
    return occ_df



# Realistic exit behavior

def _sample_visitor_exit_type(
    rng: np.random.Generator,
    early_exit_share: float = 0.10,
    loiter_share: float = 0.15,
) -> str:
    """
    Visitor groups:
    - early: exits before completing all rooms (minority)
    - complete: visits all rooms then exits (majority)
    - loiter: visits all rooms, stays longer, then exits (minority)
    """
    early_exit_share = float(np.clip(early_exit_share, 0.0, 0.40))
    loiter_share = float(np.clip(loiter_share, 0.0, 0.40))
    if early_exit_share + loiter_share > 0.90:
        loiter_share = 0.90 - early_exit_share

    u = float(rng.random())
    if u < early_exit_share:
        return "early"
    if u < early_exit_share + loiter_share:
        return "loiter"
    return "complete"


def _exit_probability_realistic(
    visitor_type: str,
    unique_visited: int,
    n_rooms: int,
    current_room: int,
) -> float:
    """
    Exit probability tuned to avoid unrealistic accumulation in the last room.

    - early visitors: can exit anytime (low but non-zero), increases with progress
    - complete visitors: almost never exit before completion, then exits quickly
    - loiter visitors: after completion, exits slower than complete
    - in last room after completing all rooms => exit probability is very high
      to prevent endless loops in Room 9
    """
    unique_visited = int(np.clip(unique_visited, 1, n_rooms))
    current_room = int(np.clip(current_room, 0, n_rooms - 1))
    progress = unique_visited / float(n_rooms)

    if visitor_type == "early":
        p = 0.03 + 0.12 * (progress ** 2)
        if unique_visited >= n_rooms:
            p = 0.75

    elif visitor_type == "complete":
        p = 0.0 if unique_visited < n_rooms else 0.92

    else:  # loiter
        p = 0.0 if unique_visited < n_rooms else 0.32

    # later rooms slightly higher exit tendency
    if n_rooms > 1:
        room_factor = 0.85 + 0.35 * (current_room / float(n_rooms - 1))
        p *= room_factor

    # hard push to exit if last room and completed
    if unique_visited >= n_rooms and current_room == (n_rooms - 1):
        p = max(p, 0.98)

    return float(np.clip(p, 0.0, 0.995))



def slice_live_window(
    occ_df: pd.DataFrame,
    end_minute: int,
    window_minutes: int,
) -> Tuple[pd.DataFrame, list]:
    start_minute = max(0, int(end_minute) - int(window_minutes))
    occ_window = occ_df[(occ_df["minute"] >= start_minute) & (occ_df["minute"] <= int(end_minute))].copy()
    room_cols = [c for c in occ_window.columns if c.startswith("Room ")]
    return occ_window, room_cols







# Routing simulation (Ordered + stochastic transitions + realistic exit)

PAPER_EDGES_FIG14 = [
    (6, 4, 0.16),
    (4, 6, 0.36),
    (4, 3, 0.21),
    (3, 4, 0.80),
    (4, 2, 0.08),
    (2, 3, 0.76),
    (3, 2, 0.20),
    (1, 2, 0.61),
    (2, 1, 0.17),
    (5, 1, 0.56),
    (1, 5, 0.39),
    (8, 5, 0.89),
    (5, 8, 0.27),
    (7, 8, 0.91),
    (8, 7, 0.11),
    (6, 7, 0.84),
    (7, 6, 0.09),
    (5, 2, 0.17),
]

def paper_transition_matrix(n_rooms: int = 9) -> np.ndarray:
    """
    Build fixed paper transition matrix P shape (n_rooms, n_rooms+1).
    Exit column kept for compat (0).
    """
    P = np.zeros((n_rooms, n_rooms + 1), dtype=float)

    for fr, to, w in PAPER_EDGES_FIG14:
        i = fr - 1
        j = to - 1
        if 0 <= i < n_rooms and 0 <= j < n_rooms:
            P[i, j] += float(w)

    for i in range(n_rooms):
        s = float(P[i, :n_rooms].sum())
        if s <= 0:
            P[i, i] = 1.0
        else:
            P[i, :n_rooms] = P[i, :n_rooms] / s
        P[i, n_rooms] = 0.0

    return P


def audioguide_transition_matrix(
    n_rooms: int = 9,
    p_next: float = 0.975,          # i -> i+1
    p_stay: float = 0.01,           # i -> i
    p_skip: float = 0.005,          # i -> i+2 (if possible)
    p_return_total: float = 0.01,   # i -> {0..i-1} (going back to previous rooms)
    return_decay: float = 0.65,     #  more likely to go back “a little” than a lot
) -> np.ndarray:

    P = np.zeros((n_rooms, n_rooms + 1), dtype=float)

    def clip01(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    p_next = clip01(p_next)
    p_stay = clip01(p_stay)
    p_skip = clip01(p_skip)
    p_return_total = clip01(p_return_total)
    return_decay = float(np.clip(return_decay, 0.05, 0.99))

    for i in range(n_rooms):
        if i == n_rooms - 1:
            # last room: self-loop (exit handled separately by your realistic exit function)
            P[i, i] = 1.0
            continue

        # main forward move
        P[i, i + 1] += p_next

        # stay
        P[i, i] += p_stay

        # skip forward
        if i + 2 < n_rooms:
            P[i, i + 2] += p_skip
        else:
            # if no i+2 exists, fold skip into i+1
            P[i, i + 1] += p_skip

        # return to previous rooms
        if i > 0 and p_return_total > 0:
            prev = np.arange(0, i)  # 0..i-1
            # weights: nearest previous gets biggest weight
            # distance d = i-1-j => weight = decay^d
            d = (i - 1) - prev
            w = (return_decay ** d).astype(float)
            w = w / float(w.sum()) if float(w.sum()) > 0 else np.ones_like(w) / len(w)
            P[i, prev] += p_return_total * w


        s = float(P[i, :n_rooms].sum())
        if s <= 0:
            P[i, i + 1] = 1.0
        else:
            P[i, :n_rooms] = P[i, :n_rooms] / s


        P[i, n_rooms] = 0.0

    return P


def simulate_museum_routing(
    arrival_rate: float = 2.0,
    simulation_time: int = 120,
    n_rooms: int = 9,
    audio_minutes_per_room: Optional[np.ndarray] = None,
    weibull_shape_k: float = 1.6,
    base_min_stay: float = 1.0,
    dwell_multiplier: float = 1.3,
    transition_matrix: Optional[np.ndarray] = None,
    start_probs: Optional[np.ndarray] = None,
    max_rooms_per_visitor: int = 25,
    travel_time_min: float = 0.0,
    seed: int = 42,
    arrival_times: Optional[np.ndarray] = None,
    # NEW: compliance split
    p_audio_compliant: float = 0.97,
    force_start_room1: bool = True,
    # Optional override for audio matrix
    audio_P: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)

    audio_minutes_per_room = _validate_audio(audio_minutes_per_room, n_rooms)
    mean_stay_per_room = _compute_mean_stay(audio_minutes_per_room, base_min_stay, dwell_multiplier)


    if start_probs is None:
        start_probs = np.zeros(n_rooms, dtype=float)
        start_probs[0] = 0.97
        if n_rooms > 1:
            start_probs[1] = 0.03
    else:
        start_probs = np.asarray(start_probs, dtype=float)
        start_probs = np.nan_to_num(start_probs, nan=0.0, posinf=0.0, neginf=0.0)
        start_probs = np.clip(start_probs, 0.0, None)
        s = float(start_probs.sum())
        start_probs = (start_probs / s) if (np.isfinite(s) and s > 0) else np.full(n_rooms, 1.0 / n_rooms)


    if transition_matrix is None:
        P_paper = paper_transition_matrix(n_rooms)
    else:
        P_paper = np.asarray(transition_matrix, dtype=float)
        if P_paper.shape != (n_rooms, n_rooms + 1):
            raise ValueError(f"transition_matrix must be shape ({n_rooms}, {n_rooms+1})")

        P_paper = np.nan_to_num(P_paper, nan=0.0, posinf=0.0, neginf=0.0)
        P_paper = np.clip(P_paper, 0.0, None)
        for i in range(n_rooms):
            s = float(P_paper[i, :n_rooms].sum())
            if (not np.isfinite(s)) or s <= 0:
                P_paper[i, :] = 0.0
                P_paper[i, i] = 1.0
            else:
                P_paper[i, :n_rooms] = P_paper[i, :n_rooms] / s
            P_paper[i, n_rooms] = 0.0


    if audio_P is None:
        P_audio = audioguide_transition_matrix(n_rooms=n_rooms)
    else:
        P_audio = np.asarray(audio_P, dtype=float)
        if P_audio.shape != (n_rooms, n_rooms + 1):
            raise ValueError(f"audio_P must be shape ({n_rooms}, {n_rooms+1})")
        P_audio = np.nan_to_num(P_audio, nan=0.0, posinf=0.0, neginf=0.0)
        P_audio = np.clip(P_audio, 0.0, None)
        for i in range(n_rooms):
            s = float(P_audio[i, :n_rooms].sum())
            if (not np.isfinite(s)) or s <= 0:
                P_audio[i, :] = 0.0
                P_audio[i, i] = 1.0
            else:
                P_audio[i, :n_rooms] = P_audio[i, :n_rooms] / s
            P_audio[i, n_rooms] = 0.0

    p_audio_compliant = float(np.clip(p_audio_compliant, 0.0, 1.0))

    # Arrivals
    if arrival_times is None:
        n_arrivals = rng.poisson(arrival_rate * simulation_time)
        arrival_times = np.sort(rng.uniform(0, simulation_time, n_arrivals))
    else:
        arrival_times = np.asarray(arrival_times, dtype=float)
        arrival_times = arrival_times[np.isfinite(arrival_times)]
        arrival_times = arrival_times[(arrival_times >= 0.0) & (arrival_times < float(simulation_time))]
        arrival_times = np.sort(arrival_times)

    rows = []
    visitor_id = 0

    for t0 in arrival_times:
        if t0 >= simulation_time:
            break

        is_compliant = (rng.random() < p_audio_compliant)
        P_used = P_audio if is_compliant else P_paper

        # Start room
        if force_start_room1:
            current = 0  # Room 1

            if rng.random() > 0.97:
                current = int(rng.choice(np.arange(n_rooms), p=start_probs))
        else:
            current = int(rng.choice(np.arange(n_rooms), p=start_probs))

        current_time = float(t0)

        visitor_type = _sample_visitor_exit_type(rng, early_exit_share=0.10, loiter_share=0.15)
        visited = np.zeros(n_rooms, dtype=bool)
        visited[current] = True

        for step in range(max_rooms_per_visitor):
            scale = _weibull_scale_for_mean(mean_stay_per_room[current], weibull_shape_k)
            dwell = float(rng.weibull(weibull_shape_k) * scale)
            exit_time = min(current_time + dwell, simulation_time)

            rows.append(
                {
                    "visitor_id": visitor_id,
                    "step": step,
                    "room_index": current,
                    "room": f"Room {current+1}",
                    "entry_time": current_time,
                    "exit_time": exit_time,
                    "stay_duration": exit_time - current_time,
                    "routing_type": "audio_compliant" if is_compliant else "non_compliant",
                }
            )

            current_time = exit_time
            if current_time >= simulation_time:
                break

            if travel_time_min > 0:
                current_time = min(current_time + travel_time_min, simulation_time)
                if current_time >= simulation_time:
                    break

            # Exit decision
            unique_visited = int(visited.sum())
            p_exit = _exit_probability_realistic(
                visitor_type=visitor_type,
                unique_visited=unique_visited,
                n_rooms=n_rooms,
                current_room=current,
            )
            if rng.random() < p_exit:
                break

            # choose next room using the visitor's fixed matrix
            p_rooms = np.asarray(P_used[current, :n_rooms], dtype=float)
            p_rooms = np.nan_to_num(p_rooms, nan=0.0, posinf=0.0, neginf=0.0)
            p_rooms = np.clip(p_rooms, 0.0, None)
            s_rooms = float(p_rooms.sum())
            if (not np.isfinite(s_rooms)) or s_rooms <= 0:
                next_room = current
            else:
                p_rooms = p_rooms / s_rooms
                next_room = int(rng.choice(np.arange(n_rooms), p=p_rooms))

            current = next_room
            visited[current] = True

        visitor_id += 1

    visits_df = pd.DataFrame(rows)
    if visits_df.empty:
        visits_df = pd.DataFrame(
            columns=["visitor_id", "step", "room_index", "room", "entry_time", "exit_time", "stay_duration", "routing_type"]
        )

    occ_df = _occupancy_from_intervals(visits_df, n_rooms, simulation_time)
    # return the paper matrix for compatibility
    return visits_df, occ_df, mean_stay_per_room, P_paper



SEED_BASE = 42
AUDIO_PRESET_OPTIONS = ["5", "6", "7", "Custom"]


def daterange(d0: date, d1: date) -> Iterable[date]:
    cur = d0
    while cur <= d1:
        yield cur
        cur = cur + timedelta(days=1)



def make_audio_df(n_rooms: int, default_minutes: float = 6.0) -> pd.DataFrame:
    default_preset = "6"
    if str(int(default_minutes)) in ("5", "6", "7"):
        default_preset = str(int(default_minutes))
    return pd.DataFrame(
        {
            "Room": [f"Room {i+1}" for i in range(n_rooms)],
            "Audio preset": [default_preset] * n_rooms,
            "Custom minutes": [float(default_minutes)] * n_rooms,
        }
    )


def validate_audio_df(df: pd.DataFrame, n_rooms: int) -> pd.DataFrame:
    df = df.copy()

    if "Room" not in df.columns:
        df["Room"] = [f"Room {i+1}" for i in range(len(df))]
    if "Audio preset" not in df.columns:
        df["Audio preset"] = "6"
    if "Custom minutes" not in df.columns:
        df["Custom minutes"] = 6.0

    df = df[["Room", "Audio preset", "Custom minutes"]].head(n_rooms)

    if df.shape[0] < n_rooms:
        missing = n_rooms - df.shape[0]
        extra = make_audio_df(missing, default_minutes=6.0)
        start = df.shape[0]
        extra["Room"] = [f"Room {i+1}" for i in range(start, n_rooms)]
        df = pd.concat([df, extra], ignore_index=True)

    df["Room"] = [f"Room {i+1}" for i in range(n_rooms)]

    df["Audio preset"] = df["Audio preset"].astype(str).apply(lambda x: x if x in AUDIO_PRESET_OPTIONS else "6")
    df["Custom minutes"] = pd.to_numeric(df["Custom minutes"], errors="coerce").fillna(6.0)
    df["Custom minutes"] = df["Custom minutes"].clip(lower=0.5, upper=30.0)

    return df


def audio_df_to_totals(df: pd.DataFrame) -> np.ndarray:
    totals: List[float] = []
    for _, row in df.iterrows():
        preset = str(row["Audio preset"])
        custom = float(row["Custom minutes"])
        if preset in ("5", "6", "7"):
            totals.append(float(preset))
        else:
            totals.append(custom)
    return np.array(totals, dtype=float)


def summarize_audio_plan(audio_df: pd.DataFrame) -> Dict[str, float]:

    totals = audio_df_to_totals(audio_df)
    return {
        "avg_audio_min": float(np.mean(totals)) if len(totals) else 0.0,
        "total_audio_min": float(np.sum(totals)) if len(totals) else 0.0,
    }



# Thresholds

def make_threshold_df(n_rooms: int, default_thr: int) -> pd.DataFrame:
    return pd.DataFrame({"Room": [f"Room {i+1}" for i in range(n_rooms)], "Threshold": [int(default_thr)] * n_rooms})


def validate_threshold_df(df: pd.DataFrame, n_rooms: int, default_thr: int) -> pd.DataFrame:
    df = df.copy()

    if "Room" not in df.columns:
        df["Room"] = [f"Room {i+1}" for i in range(len(df))]
    if "Threshold" not in df.columns:
        df["Threshold"] = default_thr

    df = df[["Room", "Threshold"]].head(n_rooms)

    if df.shape[0] < n_rooms:
        missing = n_rooms - df.shape[0]
        extra = make_threshold_df(missing, default_thr)
        start = df.shape[0]
        extra["Room"] = [f"Room {i+1}" for i in range(start, n_rooms)]
        df = pd.concat([df, extra], ignore_index=True)

    df["Room"] = [f"Room {i+1}" for i in range(n_rooms)]
    df["Threshold"] = pd.to_numeric(df["Threshold"], errors="coerce").fillna(default_thr).astype(int)
    df["Threshold"] = df["Threshold"].clip(lower=1, upper=500)
    return df


def build_threshold_series(
    mode: str,
    n_rooms: int,
    global_thr: int,
    advanced_thr_df: Optional[pd.DataFrame] = None,
) -> pd.Series:
    if mode.startswith("Advanced") and advanced_thr_df is not None and not advanced_thr_df.empty:
        df = validate_threshold_df(advanced_thr_df, n_rooms, global_thr)
        return pd.Series(df["Threshold"].to_numpy(dtype=int), index=df["Room"].tolist())
    return pd.Series({f"Room {i+1}": int(global_thr) for i in range(n_rooms)})



# ARRIVALS: Hourly profile

def _hours_covered(open_minute: int, close_minute: int) -> List[int]:
    minutes = np.arange(open_minute, close_minute)
    return sorted(set((minutes // 60).tolist()))


def default_hourly_profile(kind: str, open_minute: int, close_minute: int) -> pd.DataFrame:
    hours = _hours_covered(open_minute, close_minute)

    if kind == "weekend":
        base = {"open": 0.9, "mid_morning": 1.3, "midday_peak": 1.6, "afternoon": 1.2, "late": 0.8}
    else:
        base = {"open": 0.7, "mid_morning": 1.1, "midday_peak": 1.4, "afternoon": 1.0, "late": 0.7}

    rows = []
    for h in hours:
        if h <= hours[0] + 1:
            mult = base["open"]
        elif h <= hours[0] + 3:
            mult = base["mid_morning"]
        elif h <= hours[0] + 5:
            mult = base["midday_peak"]
        elif h <= hours[0] + 7:
            mult = base["afternoon"]
        else:
            mult = base["late"]
        rows.append({"Hour": int(h), "Multiplier": float(mult)})

    return pd.DataFrame(rows)


def validate_hourly_profile_df(df: pd.DataFrame, open_minute: int, close_minute: int) -> pd.DataFrame:
    df = df.copy()
    hours_needed = _hours_covered(open_minute, close_minute)

    if "Hour" not in df.columns:
        df["Hour"] = hours_needed
    if "Multiplier" not in df.columns:
        df["Multiplier"] = 1.0

    df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce").fillna(0).astype(int)
    df["Multiplier"] = pd.to_numeric(df["Multiplier"], errors="coerce").fillna(1.0).astype(float)

    df = df[df["Hour"].isin(hours_needed)].copy().drop_duplicates(subset=["Hour"], keep="first")

    existing = set(df["Hour"].tolist())
    for h in hours_needed:
        if h not in existing:
            df = pd.concat([df, pd.DataFrame([{"Hour": h, "Multiplier": 1.0}])], ignore_index=True)

    df["Multiplier"] = df["Multiplier"].clip(lower=0.0, upper=5.0)
    return df.sort_values("Hour").reset_index(drop=True)


def hourly_profile_to_map(df: pd.DataFrame) -> Dict[int, float]:
    out = {}
    for _, row in df.iterrows():
        out[int(row["Hour"])] = float(row["Multiplier"])
    return out


def build_lambda_by_minute(
    base_arrival_rate: float,
    open_minute: int,
    close_minute: int,
    hourly_multiplier_map: Dict[int, float],
    normalize: bool = True,
) -> np.ndarray:
    daily_minutes = close_minute - open_minute
    minutes_of_day = open_minute + np.arange(daily_minutes)
    hours = (minutes_of_day // 60).astype(int)

    mult = np.array([hourly_multiplier_map.get(int(h), 1.0) for h in hours], dtype=float)
    mult = np.clip(mult, 0.0, None)

    if normalize:
        m = float(np.mean(mult)) if len(mult) else 1.0
        if np.isfinite(m) and m > 0:
            mult = mult / m

    return float(base_arrival_rate) * mult



# ARRIVALS: Piecewise profile

def make_piecewise_profile_df(open_minute: int, close_minute: int, kind: str = "weekday") -> pd.DataFrame:

    open_h = int(open_minute // 60)
    close_h = int(math.ceil(close_minute / 60))

    # defaults: 4 intervals
    h1 = open_h
    h2 = min(open_h + 2, close_h)
    h3 = min(open_h + 5, close_h)
    h4 = min(open_h + 8, close_h)
    h5 = close_h

    if kind == "weekend":
        m1, m2, m3, m4 = 0.95, 1.35, 1.55, 0.85
    else:
        m1, m2, m3, m4 = 0.75, 1.15, 1.45, 0.75

    rows = []
    for a, b, m in [(h1, h2, m1), (h2, h3, m2), (h3, h4, m3), (h4, h5, m4)]:
        if b > a:
            rows.append({"Start hour": int(a), "End hour": int(b), "Multiplier": float(m)})

    return pd.DataFrame(rows)


def validate_piecewise_profile_df(df: pd.DataFrame, open_minute: int, close_minute: int) -> pd.DataFrame:
    df = df.copy()

    open_h = int(open_minute // 60)
    close_h = int(math.ceil(close_minute / 60))

    if "Start hour" not in df.columns:
        df["Start hour"] = open_h
    if "End hour" not in df.columns:
        df["End hour"] = close_h
    if "Multiplier" not in df.columns:
        df["Multiplier"] = 1.0

    df["Start hour"] = pd.to_numeric(df["Start hour"], errors="coerce").fillna(open_h).astype(int)
    df["End hour"] = pd.to_numeric(df["End hour"], errors="coerce").fillna(close_h).astype(int)
    df["Multiplier"] = pd.to_numeric(df["Multiplier"], errors="coerce").fillna(1.0).astype(float)

    df["Start hour"] = df["Start hour"].clip(lower=open_h, upper=close_h)
    df["End hour"] = df["End hour"].clip(lower=open_h, upper=close_h)
    df["Multiplier"] = df["Multiplier"].clip(lower=0.0, upper=5.0)

    # keep only valid segments
    df = df[df["End hour"] > df["Start hour"]].copy()
    df = df.sort_values(["Start hour", "End hour"]).reset_index(drop=True)

    if df.empty:
        return make_piecewise_profile_df(open_minute, close_minute, kind="weekday")

    return df


def build_lambda_by_minute_piecewise(
    base_arrival_rate: float,
    open_minute: int,
    close_minute: int,
    piecewise_df: pd.DataFrame,
    normalize: bool = True,
) -> np.ndarray:
    piecewise_df = validate_piecewise_profile_df(piecewise_df, open_minute, close_minute)

    daily_minutes = int(close_minute - open_minute)
    minutes_of_day = open_minute + np.arange(daily_minutes)
    hours = (minutes_of_day // 60).astype(int)

    mult = np.ones(daily_minutes, dtype=float)

    # last interval wins if overlaps
    for _, row in piecewise_df.iterrows():
        a = int(row["Start hour"])
        b = int(row["End hour"])
        m = float(row["Multiplier"])
        mask = (hours >= a) & (hours < b)
        mult[mask] = m

    mult = np.clip(mult, 0.0, None)

    if normalize:
        mean_mult = float(np.mean(mult)) if len(mult) else 1.0
        if np.isfinite(mean_mult) and mean_mult > 0:
            mult = mult / mean_mult

    return float(base_arrival_rate) * mult


def sample_arrival_times_inhomogeneous(lam_by_minute: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    arrivals: List[float] = []
    for m, lam in enumerate(lam_by_minute):
        if lam <= 0:
            continue
        n = int(rng.poisson(lam))
        if n > 0:
            arrivals.extend((m + rng.random(n)).tolist())

    return np.sort(np.array(arrivals, dtype=float)) if arrivals else np.array([], dtype=float)



# KPI computation

def compute_run_kpis(
    occ_df: pd.DataFrame,
    thr_series: pd.Series,
    default_thr: int,
    open_minute: int,
    close_minute: int,
) -> pd.DataFrame:
    df = occ_df.copy()
    df = df[(df["minute"] >= open_minute) & (df["minute"] <= close_minute)]
    room_cols = [c for c in df.columns if c.startswith("Room ")]

    if df.empty or len(room_cols) == 0:
        return pd.DataFrame()

    occ = df[room_cols].astype(float)
    thr = thr_series.reindex(room_cols).fillna(default_thr).astype(float)

    above = occ.gt(thr, axis=1)
    minutes_above = above.sum(axis=0)

    out = pd.DataFrame(
        {
            "Room": room_cols,
            "Threshold": thr.values.astype(int),
            "Avg occupancy": occ.mean(axis=0).round(2).values,
            "Max occupancy": occ.max(axis=0).values.astype(int),
            "Minutes above thr": minutes_above.values.astype(int),
            "% time above thr": (minutes_above / len(df) * 100).round(2).values,
        }
    )
    out["Any exceed?"] = out["Minutes above thr"] > 0
    return out


def aggregate_period_kpis(kpis_list: List[pd.DataFrame]) -> pd.DataFrame:
    if not kpis_list:
        return pd.DataFrame()

    big = pd.concat(kpis_list, ignore_index=True)
    if big.empty:
        return pd.DataFrame()

    def p95(x):
        return float(np.percentile(np.asarray(x, dtype=float), 95))

    g = big.groupby(["Room"], as_index=False)
    out = g.agg(
        {
            "Threshold": "first",
            "Avg occupancy": "mean",
            "Max occupancy": ["mean", p95],
            "Minutes above thr": ["mean", p95],
            "% time above thr": ["mean", p95],
            "Any exceed?": "mean",
        }
    )

    out.columns = [
        "Room",
        "Threshold",
        "Avg occupancy (mean)",
        "Max occupancy (mean)",
        "Max occupancy (P95)",
        "Minutes above thr (mean)",
        "Minutes above thr (P95)",
        "% time above thr (mean)",
        "% time above thr (P95)",
        "P(exceed) per run",
    ]
    out["P(exceed) per run"] = (out["P(exceed) per run"] * 100).round(1)

    out = out.sort_values(
        by=["Minutes above thr (mean)", "Max occupancy (P95)"],
        ascending=False,
    ).reset_index(drop=True)

    return out


def summarize_overall(period_kpis: pd.DataFrame) -> Dict[str, int | str]:
    if period_kpis.empty:
        return {"worst_room": "N/A", "total_overload_mean": 0, "max_occ_p95": 0}

    worst_room = str(period_kpis.iloc[0]["Room"])
    total_overload_mean = int(round(period_kpis["Minutes above thr (mean)"].sum()))
    max_occ_p95 = int(np.max(period_kpis["Max occupancy (P95)"].to_numpy(dtype=float)))

    return {
        "worst_room": worst_room,
        "total_overload_mean": total_overload_mean,
        "max_occ_p95": max_occ_p95,
    }



# Monte Carlo Planning (Routing)

def run_period_monte_carlo_routing(
    d0: date,
    d1: date,
    n_runs_per_day: int,
    base_arrival_rate: float,
    open_minute: int,
    close_minute: int,
    audio_totals: np.ndarray,
    weibull_k: float,
    base_min_stay: float,
    dwell_mult: float,
    thr_series: pd.Series,
    default_thr: int,
    normalize_profile: bool = True,
    max_store_runs_for_bands: int = 1000,
    # NEW arrivals controls
    weekday_piecewise_df: Optional[pd.DataFrame] = None,
    weekend_piecewise_df: Optional[pd.DataFrame] = None,
    # NEW routing compliance split
    p_audio_compliant: float = 0.97,
) -> Tuple[
    pd.DataFrame,
    Dict[str, int | str],
    Optional[pd.DataFrame],
    Optional[date],
    Dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    if n_runs_per_day <= 0:
        empty = pd.DataFrame()
        return empty, summarize_overall(empty), None, None, {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    n_rooms = 9
    daily_minutes = close_minute - open_minute

    # validate arrival profiles

    if weekday_piecewise_df is None:
        weekday_piecewise_df = make_piecewise_profile_df(open_minute, close_minute, kind="weekday")
    if weekend_piecewise_df is None:
        weekend_piecewise_df = make_piecewise_profile_df(open_minute, close_minute, kind="weekend")

    weekday_piecewise_df = validate_piecewise_profile_df(weekday_piecewise_df, open_minute, close_minute)
    weekend_piecewise_df = validate_piecewise_profile_df(weekend_piecewise_df, open_minute, close_minute)

    # KPI aggregation
    kpis_runs: List[pd.DataFrame] = []
    example_occ_df = None
    example_day = None

    # heatmaps: excess + exceed probability
    room_cols = [f"Room {i+1}" for i in range(n_rooms)]
    thr_vec = thr_series.reindex(room_cols).fillna(default_thr).to_numpy(dtype=float)

    excess_sum_all = np.zeros((daily_minutes + 1, n_rooms), dtype=float)
    exceed_sum_all = np.zeros((daily_minutes + 1, n_rooms), dtype=float)
    count_all = 0

    excess_sum_wd = np.zeros((daily_minutes + 1, n_rooms), dtype=float)
    exceed_sum_wd = np.zeros((daily_minutes + 1, n_rooms), dtype=float)
    count_wd = 0

    excess_sum_we = np.zeros((daily_minutes + 1, n_rooms), dtype=float)
    exceed_sum_we = np.zeros((daily_minutes + 1, n_rooms), dtype=float)
    count_we = 0

    # diagnostics: mean entries per run
    entries_sum = np.zeros(n_rooms, dtype=float)
    entries_count = 0

    # occupancy bands sampling
    max_store_runs_for_bands = int(max(0, max_store_runs_for_bands))
    reservoir: List[np.ndarray] = []
    rng_reservoir = np.random.default_rng(SEED_BASE + 999)
    run_index = 0

    # expected mean stay table
    mean_stay_per_room = _compute_mean_stay(audio_totals, base_min_stay, dwell_mult)

    days = list(daterange(d0, d1))
    for di, day in enumerate(days):
        is_weekend = (day.weekday() >= 5)

        for r in range(int(n_runs_per_day)):
            seed = int(SEED_BASE + di * 10_000 + r)
            rng = np.random.default_rng(seed)

            # arrivals
            piece_df = weekend_piecewise_df if is_weekend else weekday_piecewise_df

            lam = build_lambda_by_minute_piecewise(
                base_arrival_rate=base_arrival_rate,
                open_minute=open_minute,
                close_minute=close_minute,
                piecewise_df=piece_df,
                normalize=normalize_profile,
            )

            arrival_times = sample_arrival_times_inhomogeneous(lam, rng)

            # routing
            visits_df, occ_df, _mean_stay, _P_used = simulate_museum_routing(
                arrival_rate=base_arrival_rate,
                simulation_time=daily_minutes,
                n_rooms=n_rooms,
                audio_minutes_per_room=audio_totals,
                weibull_shape_k=weibull_k,
                base_min_stay=base_min_stay,
                dwell_multiplier=dwell_mult,
                transition_matrix=None,
                seed=seed,
                arrival_times=arrival_times,
                p_audio_compliant=float(p_audio_compliant),
                force_start_room1=True,
            )


            if not occ_df.empty:
                occ_mat = occ_df[room_cols].to_numpy(dtype=float)

                excess_mat = np.maximum(0.0, occ_mat - thr_vec[None, :])

                exceed_mat = (occ_mat > thr_vec[None, :]).astype(float)

                excess_sum_all += excess_mat
                exceed_sum_all += exceed_mat
                count_all += 1

                if is_weekend:
                    excess_sum_we += excess_mat
                    exceed_sum_we += exceed_mat
                    count_we += 1
                else:
                    excess_sum_wd += excess_mat
                    exceed_sum_wd += exceed_mat
                    count_wd += 1


                if max_store_runs_for_bands > 0:
                    if len(reservoir) < max_store_runs_for_bands:
                        reservoir.append(occ_mat.astype(np.float32))
                    else:
                        j = int(rng_reservoir.integers(0, run_index + 1))
                        if j < max_store_runs_for_bands:
                            reservoir[j] = occ_mat.astype(np.float32)

                run_index += 1


            if not visits_df.empty:
                vc = visits_df["room"].value_counts()
                for i in range(n_rooms):
                    entries_sum[i] += float(vc.get(f"Room {i+1}", 0.0))
                entries_count += 1


            occ_df_shifted = occ_df.copy()
            occ_df_shifted["minute"] = occ_df_shifted["minute"] + open_minute

            k = compute_run_kpis(
                occ_df=occ_df_shifted,
                thr_series=thr_series,
                default_thr=default_thr,
                open_minute=open_minute,
                close_minute=close_minute,
            )
            if not k.empty:
                k["Day"] = str(day)
                k["Run"] = r
                kpis_runs.append(k)

            if example_occ_df is None:
                example_occ_df = occ_df_shifted.copy()
                example_day = day

    period_kpis = aggregate_period_kpis(kpis_runs)
    overall = summarize_overall(period_kpis)


    minutes_real = open_minute + np.arange(daily_minutes + 1)

    def _to_df(mat: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(mat, columns=room_cols)
        df.insert(0, "minute", minutes_real)
        return df

    def _cond_mean(excess_sum: np.ndarray, exceed_sum: np.ndarray) -> np.ndarray:
        out = np.zeros_like(excess_sum, dtype=float)
        mask = exceed_sum > 0
        out[mask] = excess_sum[mask] / exceed_sum[mask]
        return out

    if count_all > 0:
        prob_all = exceed_sum_all / float(count_all)  # 0..1 (ok)
        excess_all = _cond_mean(excess_sum_all, exceed_sum_all)
    else:
        prob_all = np.zeros_like(exceed_sum_all)
        excess_all = np.zeros_like(excess_sum_all)

    if count_wd > 0:
        prob_wd = exceed_sum_wd / float(count_wd)
        excess_wd = _cond_mean(excess_sum_wd, exceed_sum_wd)
    else:
        prob_wd = np.zeros_like(exceed_sum_wd)
        excess_wd = np.zeros_like(excess_sum_wd)

    if count_we > 0:
        prob_we = exceed_sum_we / float(count_we)
        excess_we = _cond_mean(excess_sum_we, exceed_sum_we)
    else:
        prob_we = np.zeros_like(exceed_sum_we)
        excess_we = np.zeros_like(excess_sum_we)


    heatmaps = {
        "excess_all": _to_df(excess_all),
        "prob_all": _to_df(prob_all),
        "excess_weekday": _to_df(excess_wd),
        "prob_weekday": _to_df(prob_wd),
        "excess_weekend": _to_df(excess_we),
        "prob_weekend": _to_df(prob_we),
    }


    # occupancy bands
    bands_rooms_df = pd.DataFrame()
    bands_total_df = pd.DataFrame()
    if len(reservoir) > 0:
        stack = np.stack(reservoir, axis=0)  # (S, T+1, rooms)
        p50 = np.percentile(stack, 50, axis=0)
        p90 = np.percentile(stack, 90, axis=0)

        cols = {}
        for i in range(n_rooms):
            cols[f"Room {i+1} P50"] = p50[:, i]
            cols[f"Room {i+1} P90"] = p90[:, i]

        bands_rooms_df = pd.DataFrame(cols)
        bands_rooms_df.insert(0, "minute", minutes_real)

        total = stack.sum(axis=2)  # (S, T+1)
        total_p50 = np.percentile(total, 50, axis=0)
        total_p90 = np.percentile(total, 90, axis=0)

        bands_total_df = pd.DataFrame({"minute": minutes_real, "Total P50": total_p50, "Total P90": total_p90})

    # diagnostics table
    mean_entries = entries_sum / float(entries_count) if entries_count > 0 else np.zeros(n_rooms)
    diagnostics_df = pd.DataFrame(
        {
            "Room": room_cols,
            "Mean stay (min)": mean_stay_per_room.round(2),
            "Mean entries per run": np.round(mean_entries, 2),
        }
    )

    return period_kpis, overall, example_occ_df, example_day, heatmaps, bands_rooms_df, bands_total_df, diagnostics_df



# Planning recommendation

def propose_audio_adjustment_period(
    audio_df: pd.DataFrame,
    period_kpis: pd.DataFrame,
    top_k_rooms: int = 3,
    step_minutes: float = 1.0,
    min_audio: float = 3.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if period_kpis.empty:
        return audio_df.copy(), pd.DataFrame()

    candidate = period_kpis.copy()
    candidate = candidate[candidate["Minutes above thr (mean)"].astype(float) > 0].copy()
    if candidate.empty:
        return audio_df.copy(), pd.DataFrame()

    new_df = audio_df.copy()
    top_rooms = candidate.head(top_k_rooms)["Room"].tolist()

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



def minute_to_hhmm(m: int) -> str:
    return f"{m//60:02d}:{m%60:02d}"