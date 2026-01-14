import numpy as np
import pandas as pd
import math


def _weibull_scale_for_mean(mean: float, shape_k: float) -> float:
    """
    For Weibull(k, scale), mean = scale * Gamma(1 + 1/k)
    => scale = mean / Gamma(1 + 1/k)
    """
    return mean / math.gamma(1.0 + 1.0 / shape_k)


def simulate_museum_week1(
    arrival_rate: float = 2.0,          # visitors per minute
    simulation_time: int = 120,         # minutes
    n_rooms: int = 6,
    audio_minutes_per_room=None,        # array-like length n_rooms
    weibull_shape_k: float = 1.6,
    base_min_stay: float = 1.0,         # minutes (even if audio = 0)
    dwell_multiplier: float = 1.3,      # E[stay_r] = base + 1.3 * audio_r
    seed: int = 42
):
    """
    Week 1 demo simulation:
    - Poisson arrivals
    - Each visitor visits exactly 1 room (start room chosen uniformly)
    - Room dwell time ~ Weibull(shape_k, scale_r) with mean tied to audio length

    Returns:
      visits_df: per-visitor, per-room entry/exit
      occ_df: occupancy time series per room (minute grid)
    """
    rng = np.random.default_rng(seed)

    if audio_minutes_per_room is None:
        audio_minutes_per_room = np.full(n_rooms, 6.0, dtype=float)
    audio_minutes_per_room = np.asarray(audio_minutes_per_room, dtype=float)
    if len(audio_minutes_per_room) != n_rooms:
        raise ValueError("audio_minutes_per_room must have length n_rooms")

    # Expected stay per room (simple link to audio)
    mean_stay_per_room = base_min_stay + dwell_multiplier * audio_minutes_per_room
    mean_stay_per_room = np.clip(mean_stay_per_room, 0.5, None)

    # Arrival process: N ~ Poisson(lambda*T), arrival times uniform on [0,T] (simple)
    n_arrivals = rng.poisson(arrival_rate * simulation_time)
    arrival_times = np.sort(rng.uniform(0, simulation_time, n_arrivals))

    # Assign each visitor to one room (uniform for Week 1)
    rooms = rng.integers(0, n_rooms, size=n_arrivals)

    # Sample Weibull stay times per visitor based on their chosen room
    scales = np.array([_weibull_scale_for_mean(mean_stay_per_room[r], weibull_shape_k) for r in rooms])
    # numpy's weibull draws shape 'a' with scale=1; multiply by scale
    stay_durations = rng.weibull(weibull_shape_k, size=n_arrivals) * scales

    exit_times = arrival_times + stay_durations
    exit_times = np.minimum(exit_times, simulation_time)  # keep within horizon

    visits_df = pd.DataFrame({
        "visitor_id": np.arange(n_arrivals),
        "room": rooms,
        "arrival_time": arrival_times,
        "exit_time": exit_times,
        "stay_duration": exit_times - arrival_times
    })

    # Occupancy per minute (0..simulation_time)
    t = np.arange(0, simulation_time + 1)
    occ = np.zeros((n_rooms, len(t)), dtype=int)

    # Compute occupancy with simple loop (still fast for Week 1 sizes)
    for r in range(n_rooms):
        rr = visits_df[visits_df["room"] == r][["arrival_time", "exit_time"]].to_numpy()
        if rr.size == 0:
            continue
        a = rr[:, 0][:, None]
        e = rr[:, 1][:, None]
        # count visitors present at each minute t: a <= t < e
        occ[r, :] = np.sum((a <= t) & (t < e), axis=0)

    occ_df = pd.DataFrame(occ.T, columns=[f"Room {i+1}" for i in range(n_rooms)])
    occ_df.insert(0, "minute", t)

    return visits_df, occ_df, mean_stay_per_room

