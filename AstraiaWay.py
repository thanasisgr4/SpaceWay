"""Asteria | Launch readiness risk intelligence prototype.

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import calendar
import json
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


APP_NAME = "Asteria | Launch Readiness"
SIMULATION_COUNT = 1_000
LAUNCH_WINDOW_DAYS = 7
BASELINE_SEED = 2026
SIMULATION_SEED = 71_903
ERA5_API_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_COLUMNS = ("wind_max_kmh", "precipitation_max_mm_h", "cloud_mean_pct")
YEARS = tuple(range(2019, 2025))
MONTHS = tuple(range(1, 13))

SITES = {
    "Kourou, French Guiana (Europe's launch gateway)": {
        "latitude": 5.236,
        "longitude": -52.768,
        "country": "French Guiana",
    },
    "Cape Canaveral, United States": {
        "latitude": 28.562,
        "longitude": -80.577,
        "country": "United States",
    },
    "Vandenberg, United States": {
        "latitude": 34.742,
        "longitude": -120.572,
        "country": "United States",
    },
    "Mahia, New Zealand": {
        "latitude": -39.262,
        "longitude": 177.864,
        "country": "New Zealand",
    },
    "Tanegashima, Japan": {
        "latitude": 30.400,
        "longitude": 130.976,
        "country": "Japan",
    },
}


@dataclass(frozen=True)
class Subsystem:
    name: str
    supplier_label: str
    baseline_risk_pct: int
    duration_days: int
    delay_low_days: float
    delay_mode_days: float
    delay_high_days: float
    dependencies: Tuple[str, ...]


class WeatherDataError(RuntimeError):
    """Raised when the historical weather source cannot provide usable data."""


def generate_subsystems() -> List[Subsystem]:
    """Create reproducible, explicitly synthetic supplier-planning inputs."""
    rng = np.random.default_rng(BASELINE_SEED)
    definitions = [
        ("1st Stage Engine", 25, 2.0, 6.0, 14.0, ()),
        ("Avionics", 18, 1.0, 4.0, 10.0, ()),
        ("Ground Support Equipment", 14, 2.0, 5.0, 12.0, ()),
        ("Propellant", 10, 1.0, 3.0, 8.0, ("Ground Support Equipment",)),
        ("Payload Integration", 12, 2.0, 5.0, 11.0, ("1st Stage Engine", "Avionics")),
    ]
    risks = rng.integers(8, 36, size=len(definitions))
    supplier_labels = [f"Illustrative supplier {chr(ord('A') + i)}" for i in range(5)]
    return [
        Subsystem(
            name=name,
            supplier_label=supplier_labels[index],
            baseline_risk_pct=int(risks[index]),
            duration_days=duration,
            delay_low_days=low,
            delay_mode_days=mode,
            delay_high_days=high,
            dependencies=dependencies,
        )
        for index, (name, duration, low, mode, high, dependencies) in enumerate(definitions)
    ]


SUBSYSTEMS = generate_subsystems()
SUBSYSTEM_BY_NAME = {subsystem.name: subsystem for subsystem in SUBSYSTEMS}


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_era5_month(
    latitude: float,
    longitude: float,
    year: int,
    month: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Fetch a month in short requests, then summarize ERA5 fields by UTC day."""
    first_day = date(year, month, 1)
    last_day = date(year, month, calendar.monthrange(year, month)[1])
    chunks = []
    chunk_start = first_day
    while chunk_start <= last_day:
        chunk_end = min(chunk_start + timedelta(days=6), last_day)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)

    def fetch_chunk(start: date, end: date) -> Tuple[pd.DataFrame, Dict[str, float]]:
        query = urlencode(
            {
                "latitude": f"{latitude:.4f}",
                "longitude": f"{longitude:.4f}",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "hourly": "wind_speed_10m,precipitation,cloud_cover",
                "timezone": "UTC",
                "wind_speed_unit": "kmh",
                "models": "era5",
            }
        )
        request = Request(
            f"{ERA5_API_URL}?{query}",
            headers={"User-Agent": "Asteria-Launch-Readiness-Prototype/1.0"},
        )
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise WeatherDataError(
                f"The weather service returned HTTP {exc.code}."
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise WeatherDataError(
                "The historical weather service could not be reached. "
                "Check the connection and try again."
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WeatherDataError(
                "The historical weather service returned an unreadable response."
            ) from exc

        try:
            hourly = payload["hourly"]
            frame = pd.DataFrame(
                {
                    "time": pd.to_datetime(hourly["time"], utc=True, errors="coerce"),
                    "wind_max_kmh": pd.to_numeric(
                        hourly["wind_speed_10m"], errors="coerce"
                    ),
                    "precipitation_max_mm_h": pd.to_numeric(
                        hourly["precipitation"], errors="coerce"
                    ),
                    "cloud_mean_pct": pd.to_numeric(
                        hourly["cloud_cover"], errors="coerce"
                    ),
                }
            ).dropna(subset=["time", *WEATHER_COLUMNS])
            if frame.empty:
                raise ValueError("No complete hourly observations were returned.")
            grid = {
                "latitude": float(payload["latitude"]),
                "longitude": float(payload["longitude"]),
                "elevation": float(payload.get("elevation", 0.0)),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise WeatherDataError(
                f"Historical weather data could not be processed: {exc}"
            ) from exc
        return frame, grid

    results = [fetch_chunk(start, end) for start, end in chunks]

    hourly_data = pd.concat([frame for frame, _ in results], ignore_index=True)
    hourly_data["date"] = hourly_data["time"].dt.date
    daily = (
        hourly_data.groupby("date", as_index=False)
        .agg(
            wind_max_kmh=("wind_max_kmh", "max"),
            precipitation_max_mm_h=("precipitation_max_mm_h", "max"),
            cloud_mean_pct=("cloud_mean_pct", "mean"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    if len(daily) < LAUNCH_WINDOW_DAYS:
        raise WeatherDataError("Too few daily observations were returned.")
    return daily, results[0][1]


def screen_weather_days(
    daily: pd.DataFrame,
    max_wind_kmh: float,
    max_precipitation_mm_h: float,
    max_cloud_pct: float,
) -> pd.DataFrame:
    """Apply editable screening thresholds; these are not launch safety limits."""
    result = daily.copy()
    result["screening_no_go"] = (
        (result["wind_max_kmh"] > max_wind_kmh)
        | (result["precipitation_max_mm_h"] > max_precipitation_mm_h)
        | (result["cloud_mean_pct"] > max_cloud_pct)
    )
    return result


def make_manual_weather_sample(
    year: int,
    month: int,
    risk_pct: int,
) -> pd.DataFrame:
    """Create an explicitly synthetic daily sequence for offline demonstrations."""
    day_count = calendar.monthrange(year, month)[1]
    rng = np.random.default_rng(BASELINE_SEED + year * 100 + month)
    no_go = np.zeros(day_count, dtype=bool)
    no_go_count = int(round(day_count * risk_pct / 100.0))
    no_go[rng.choice(day_count, size=no_go_count, replace=False)] = True
    dates = pd.date_range(f"{year}-{month:02d}-01", periods=day_count, freq="D")
    return pd.DataFrame(
        {
            "date": dates.date,
            "wind_max_kmh": np.where(
                no_go, rng.uniform(36, 65, day_count), rng.uniform(5, 28, day_count)
            ),
            "precipitation_max_mm_h": np.where(
                no_go, rng.uniform(2.1, 8, day_count), rng.uniform(0, 1.5, day_count)
            ),
            "cloud_mean_pct": np.where(
                no_go, rng.uniform(81, 100, day_count), rng.uniform(10, 75, day_count)
            ),
            "screening_no_go": no_go,
        }
    )


def adjust_weather_frequency(
    historical_flags: np.ndarray,
    scenario_risk_pct: int,
    seed: int,
) -> np.ndarray:
    """Adjust baseline screening flags to a user-selected scenario frequency."""
    rng = np.random.default_rng(seed)
    flags = np.asarray(historical_flags, dtype=bool)
    baseline_rate = float(np.mean(flags))
    target_rate = scenario_risk_pct / 100.0
    if np.isclose(baseline_rate, target_rate):
        return flags.copy()
    if baseline_rate <= 0.0 or baseline_rate >= 1.0:
        return rng.random(len(flags)) < target_rate
    if target_rate < baseline_rate:
        keep_probability = target_rate / baseline_rate
        return flags & (rng.random(len(flags)) < keep_probability)
    add_probability = (target_rate - baseline_rate) / (1.0 - baseline_rate)
    added_flags = (~flags) & (rng.random(len(flags)) < add_probability)
    return flags | added_flags


def schedule_project(
    durations: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """Calculate earliest task starts and finishes from the dependency graph."""
    starts: Dict[str, float] = {}
    finishes: Dict[str, float] = {}
    for subsystem in SUBSYSTEMS:
        start = max((finishes[name] for name in subsystem.dependencies), default=0.0)
        starts[subsystem.name] = start
        finishes[subsystem.name] = start + durations[subsystem.name]
    project_finish = max(
        finishes["Propellant"],
        finishes["Payload Integration"],
    )
    return starts, finishes, project_finish


def find_critical_path(
    starts: Dict[str, float],
    finishes: Dict[str, float],
) -> set[str]:
    """Trace the longest expected-duration path through the project network."""
    terminal_tasks = ("Propellant", "Payload Integration")
    current = max(terminal_tasks, key=lambda name: finishes[name])
    path: set[str] = set()
    while current not in path:
        path.add(current)
        dependencies = SUBSYSTEM_BY_NAME[current].dependencies
        if not dependencies:
            break
        predecessor = max(dependencies, key=lambda name: finishes[name])
        if not np.isclose(finishes[predecessor], starts[current]):
            break
        current = predecessor
    return path


@st.cache_data(show_spinner=False)
def run_simulation(
    risk_percentages: Tuple[int, ...],
    scenario_weather_risk_pct: int,
    weather_flags: Tuple[bool, ...],
) -> Tuple[float, pd.DataFrame, np.ndarray, np.ndarray]:
    """Run a reproducible supplier-delay and historical-weather simulation."""
    seed_offset = scenario_weather_risk_pct + sum(
        (index + 1) * risk for index, risk in enumerate(risk_percentages)
    )
    seed_offset += sum(
        (index + 1) * int(flag) for index, flag in enumerate(weather_flags)
    )
    rng = np.random.default_rng(SIMULATION_SEED + seed_offset)
    subsystem_delays: Dict[str, np.ndarray] = {}
    subsystem_finishes: Dict[str, np.ndarray] = {}

    for subsystem, risk_pct in zip(SUBSYSTEMS, risk_percentages):
        event_occurs = rng.random(SIMULATION_COUNT) < risk_pct / 100.0
        delay_impact = rng.triangular(
            subsystem.delay_low_days,
            subsystem.delay_mode_days,
            subsystem.delay_high_days,
            size=SIMULATION_COUNT,
        )
        delays = event_occurs * delay_impact
        subsystem_delays[subsystem.name] = delays
        dependency_finish = np.zeros(SIMULATION_COUNT)
        if subsystem.dependencies:
            dependency_finish = np.maximum.reduce(
                [subsystem_finishes[name] for name in subsystem.dependencies]
            )
        subsystem_finishes[subsystem.name] = (
            dependency_finish + subsystem.duration_days + delays
        )

    baseline_finish = schedule_project(
        {subsystem.name: float(subsystem.duration_days) for subsystem in SUBSYSTEMS}
    )[2]
    simulated_finish = np.maximum(
        subsystem_finishes["Propellant"],
        subsystem_finishes["Payload Integration"],
    )
    supplier_slippage = np.maximum(simulated_finish - baseline_finish, 0.0)

    flags = np.asarray(weather_flags, dtype=bool)
    if len(flags) < LAUNCH_WINDOW_DAYS:
        raise ValueError(
            f"At least {LAUNCH_WINDOW_DAYS} daily weather observations are required."
        )
    blocks = np.stack(
        [
            flags[start : start + LAUNCH_WINDOW_DAYS]
            for start in range(len(flags) - LAUNCH_WINDOW_DAYS + 1)
        ]
    )
    selected_blocks = blocks[rng.integers(0, len(blocks), size=SIMULATION_COUNT)]
    clear_weather = ~selected_blocks
    has_clear_day = clear_weather.any(axis=1)
    first_clear_day = np.argmax(clear_weather, axis=1)
    weather_scrub_days = np.where(
        has_clear_day, first_clear_day, LAUNCH_WINDOW_DAYS
    ).astype(float)

    total_slippage = supplier_slippage + weather_scrub_days
    success_probability = float(
        np.mean(total_slippage < LAUNCH_WINDOW_DAYS) * 100.0
    )
    bottlenecks = pd.DataFrame(
        {
            "Subsystem": [subsystem.name for subsystem in SUBSYSTEMS],
            "Average Delay (days)": [
                float(np.mean(subsystem_delays[subsystem.name]))
                for subsystem in SUBSYSTEMS
            ],
        }
    ).sort_values("Average Delay (days)", ascending=False)
    return success_probability, bottlenecks, supplier_slippage, weather_scrub_days


def render_gauge(probability: float) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability,
            number={"suffix": "%", "font": {"size": 46}},
            title={"text": "Chance of readiness inside window"},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "#2B6C5A"},
                "steps": [
                    {"range": [0, 60], "color": "#F9E2E0"},
                    {"range": [60, 85], "color": "#F8EFCF"},
                    {"range": [85, 100], "color": "#DCEFE5"},
                ],
                "threshold": {
                    "line": {"color": "#25333B", "width": 3},
                    "thickness": 0.8,
                    "value": 85,
                },
            },
        )
    )
    figure.update_layout(
        height=300,
        margin={"l": 22, "r": 22, "t": 65, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#23313A", "family": "Arial"},
    )
    return figure


def render_schedule(
    risk_percentages: Tuple[int, ...],
    mean_weather_scrub: float,
) -> go.Figure:
    expected_durations = {
        subsystem.name: subsystem.duration_days
        + risk_pct
        / 100.0
        * (
            subsystem.delay_low_days
            + subsystem.delay_mode_days
            + subsystem.delay_high_days
        )
        / 3.0
        for subsystem, risk_pct in zip(SUBSYSTEMS, risk_percentages)
    }
    starts, finishes, project_finish = schedule_project(expected_durations)
    critical_path = find_critical_path(starts, finishes)
    labels = [subsystem.name for subsystem in SUBSYSTEMS]
    labels.append("Historical weather scrub (expected)")
    task_starts = [starts[name] for name in labels[:-1]] + [project_finish]
    task_durations = [expected_durations[name] for name in labels[:-1]] + [
        mean_weather_scrub
    ]
    colors = [
        "#C27D3B" if name in critical_path else "#668D9B"
        for name in labels[:-1]
    ] + ["#8171A8"]

    figure = go.Figure(
        go.Bar(
            x=task_durations,
            y=labels,
            base=task_starts,
            orientation="h",
            marker_color=colors,
            text=[f"{duration:.1f} d" for duration in task_durations],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=(
                "%{y}<br>Start: day %{base:.1f}"
                "<br>Expected duration: %{x:.1f} days<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.add_vrect(
        x0=project_finish,
        x1=project_finish + LAUNCH_WINDOW_DAYS,
        fillcolor="#DCEFE5",
        opacity=0.45,
        line_width=0,
        annotation_text="7-day launch window",
        annotation_position="top",
    )
    figure.add_vline(
        x=project_finish,
        line_dash="dash",
        line_color="#2B6C5A",
        annotation_text="Expected readiness",
        annotation_position="bottom",
    )
    figure.update_layout(
        title="Expected schedule and readiness window",
        xaxis_title="Days from project start",
        yaxis_title="",
        yaxis={"autorange": "reversed"},
        xaxis={"rangemode": "tozero", "showgrid": True, "gridcolor": "#E7ECEA"},
        height=360,
        margin={"l": 18, "r": 24, "t": 72, "b": 45},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font={"color": "#23313A", "family": "Arial"},
    )
    return figure


def render_bottlenecks(bottlenecks: pd.DataFrame) -> go.Figure:
    chart_data = bottlenecks.sort_values("Average Delay (days)", ascending=True)
    colors = [
        "#C27D3B" if index == len(chart_data) - 1 else "#8EADB6"
        for index in range(len(chart_data))
    ]
    figure = go.Figure(
        go.Bar(
            x=chart_data["Average Delay (days)"],
            y=chart_data["Subsystem"],
            orientation="h",
            marker_color=colors,
            text=[f"{value:.2f} d" for value in chart_data["Average Delay (days)"]],
            textposition="outside",
            hovertemplate="%{y}<br>Mean delay: %{x:.2f} days<extra></extra>",
        )
    )
    figure.update_layout(
        title="Mean simulated delay by subsystem",
        xaxis_title="Mean component delay (days per simulation)",
        yaxis_title="",
        height=360,
        margin={"l": 18, "r": 35, "t": 72, "b": 45},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font={"color": "#23313A", "family": "Arial"},
    )
    return figure


def render_weather_history(daily: pd.DataFrame, era5_mode: bool) -> go.Figure:
    figure = go.Figure()
    colors = [
        "#C27D3B" if bool(flag) else "#6E9D8F"
        for flag in daily["screening_no_go"]
    ]
    hover = []
    for row in daily.itertuples():
        if era5_mode:
            hover.append(
                f"Wind max: {row.wind_max_kmh:.1f} km/h"
                f"<br>Hourly precipitation max: {row.precipitation_max_mm_h:.1f} mm"
                f"<br>Mean cloud cover: {row.cloud_mean_pct:.0f}%"
                f"<br>Screen: {'Flagged' if row.screening_no_go else 'Not flagged'}"
            )
        else:
            hover.append(
                f"Illustrative daily weather scenario"
                f"<br>Screen: {'Flagged' if row.screening_no_go else 'Not flagged'}"
            )
    figure.add_trace(
        go.Bar(
            x=daily["date"].astype(str),
            y=[1 if flag else 0 for flag in daily["screening_no_go"]],
            marker_color=colors,
            customdata=hover,
            hovertemplate="%{x}<br>%{customdata}<extra></extra>",
            name="Daily screen",
        )
    )
    figure.update_layout(
        title="Historical daily weather screening" if era5_mode else "Synthetic weather scenario",
        xaxis_title="Day",
        yaxis={
            "tickmode": "array",
            "tickvals": [0, 1],
            "ticktext": ["No threshold exceeded", "One or more thresholds exceeded"],
            "range": [-0.15, 1.2],
        },
        height=300,
        margin={"l": 20, "r": 20, "t": 68, "b": 45},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font={"color": "#23313A", "family": "Arial"},
        showlegend=False,
    )
    return figure


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #F4F7F5; color: #23313A; }
        .block-container { max-width: 1440px; padding-top: 1.6rem; }
        [data-testid="stSidebar"] { background: #EDF2EF; }
        .hero {
            background: linear-gradient(120deg, #173D3B 0%, #2B6C5A 65%, #507F74 100%);
            color: white; border-radius: 18px; padding: 1.55rem 1.8rem;
            margin-bottom: 1.1rem;
        }
        .hero h1 { color: white; margin: 0 0 .35rem 0; font-size: 2rem; }
        .hero p { color: #E0EBE6; margin: 0; font-size: 1rem; }
        .eyebrow {
            text-transform: uppercase; letter-spacing: .13em; font-size: .73rem;
            color: #C5DED3; font-weight: 700; margin-bottom: .45rem;
        }
        .section-card {
            background: white; border: 1px solid #E1E9E4; border-radius: 14px;
            padding: 1rem 1.2rem; margin-bottom: .8rem;
        }
        .small-note { color: #61716B; font-size: .88rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Mission readiness intelligence · Prototype</div>
          <h1>Asteria</h1>
          <p>Explore how supplier delivery uncertainty and observed launch-site
          weather can affect a planned readiness window.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("Scenario workspace")
    st.sidebar.caption("Change inputs to explore a transparent what-if scenario.")
    weather_mode = st.sidebar.radio(
        "Weather evidence",
        ("ERA5 historical data", "Manual demo scenario"),
        help=(
            "ERA5 loads historical weather for a selected site and month. "
            "Manual mode is synthetic and works without internet access."
        ),
    )
    era5_mode = weather_mode == "ERA5 historical data"

    selected_site = st.sidebar.selectbox("Launch site", tuple(SITES.keys()))
    site = SITES[selected_site]
    year = st.sidebar.selectbox("Historical reference year", YEARS, index=YEARS.index(2024))
    month = st.sidebar.selectbox(
        "Reference month",
        MONTHS,
        index=MONTHS.index(6),
        format_func=lambda value: calendar.month_name[value],
    )

    if era5_mode:
        st.sidebar.subheader("Weather screening thresholds")
        max_wind_kmh = st.sidebar.slider(
            "Daily maximum 10 m wind (km/h)",
            min_value=10,
            max_value=100,
            value=35,
            step=5,
        )
        max_precipitation_mm_h = st.sidebar.slider(
            "Maximum hourly precipitation (mm)",
            min_value=0,
            max_value=20,
            value=2,
            step=1,
        )
        max_cloud_pct = st.sidebar.slider(
            "Mean daily cloud cover (%)",
            min_value=0,
            max_value=100,
            value=85,
            step=5,
        )
        st.sidebar.caption(
            "Illustrative screening thresholds—not official launch-weather or "
            "flight-safety criteria."
        )

    grid: Dict[str, float] = {
        "latitude": float(site["latitude"]),
        "longitude": float(site["longitude"]),
        "elevation": 0.0,
    }
    if era5_mode:
        with st.spinner(f"Loading ERA5 historical weather for {selected_site}..."):
            try:
                daily_weather, grid = fetch_era5_month(
                    float(site["latitude"]),
                    float(site["longitude"]),
                    int(year),
                    int(month),
                )
            except WeatherDataError as exc:
                st.error(f"Could not load ERA5 data: {exc}")
                st.info(
                    "Choose **Manual demo scenario** in the sidebar to continue "
                    "without live data. The manual scenario is synthetic."
                )
                st.stop()
        daily_weather = screen_weather_days(
            daily_weather,
            max_wind_kmh,
            max_precipitation_mm_h,
            max_cloud_pct,
        )
        reference_weather_risk = int(
            round(daily_weather["screening_no_go"].mean() * 100)
        )
        st.sidebar.metric("ERA5 screening rate", f"{reference_weather_risk}%")
        scenario_weather_risk = st.sidebar.slider(
            "Weather risk scenario (%)",
            min_value=0,
            max_value=100,
            value=reference_weather_risk,
            key=f"weather_risk_{selected_site}_{year}_{month}",
            help=(
                "Defaults to the selected month's ERA5 screening rate. Changing "
                "it proportionally adjusts the daily flags for what-if analysis."
            ),
        )
    else:
        scenario_weather_risk = st.sidebar.slider(
            "Synthetic weather risk scenario (%)",
            min_value=0,
            max_value=100,
            value=25,
        )
        daily_weather = make_manual_weather_sample(
            int(year), int(month), scenario_weather_risk
        )
        daily_weather["screening_no_go"] = daily_weather["screening_no_go"].astype(bool)
        reference_weather_risk = int(
            round(daily_weather["screening_no_go"].mean() * 100)
        )
        st.sidebar.warning("Manual weather data is synthetic, not observed.")

    st.sidebar.subheader("Supplier / subsystem risk")
    selected_risks: List[int] = []
    for subsystem in SUBSYSTEMS:
        selected_risks.append(
            st.sidebar.slider(
                f"{subsystem.name} delay risk (%)",
                min_value=0,
                max_value=100,
                value=subsystem.baseline_risk_pct,
                help=(
                    f"{subsystem.supplier_label}; synthetic baseline. "
                    f"Nominal duration: {subsystem.duration_days} days."
                ),
                key=f"risk_{subsystem.name}",
            )
        )
    risk_percentages = tuple(selected_risks)

    flags_seed = (
        BASELINE_SEED
        + int(year) * 101
        + int(month) * 17
        + sum(ord(character) for character in selected_site)
    )
    scenario_weather_flags = adjust_weather_frequency(
        daily_weather["screening_no_go"].to_numpy(dtype=bool),
        scenario_weather_risk,
        flags_seed,
    )
    probability, bottlenecks, supplier_slippage, weather_scrub_days = run_simulation(
        risk_percentages,
        scenario_weather_risk,
        tuple(bool(flag) for flag in scenario_weather_flags),
    )
    total_slippage = supplier_slippage + weather_scrub_days
    median_slippage = float(np.median(total_slippage))
    p90_slippage = float(np.percentile(total_slippage, 90))
    successes = int(np.sum(total_slippage < LAUNCH_WINDOW_DAYS))
    mean_weather_scrub = float(np.mean(weather_scrub_days))
    top_bottleneck = bottlenecks.iloc[0]
    weather_rate_label = "ERA5 reference" if era5_mode else "Synthetic baseline"

    st.caption(
        f"**Decision question:** Can this illustrative program reach readiness "
        f"inside a {LAUNCH_WINDOW_DAYS}-day window under the selected assumptions?"
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("Within-window runs", f"{probability:.1f}%")
    metric_columns[1].metric("Median total slippage", f"{median_slippage:.1f} days")
    metric_columns[2].metric("90th percentile slippage", f"{p90_slippage:.1f} days")
    metric_columns[3].metric(
        "Top mean component delay",
        str(top_bottleneck["Subsystem"]),
        f"{top_bottleneck['Average Delay (days)']:.2f} mean days",
    )

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.plotly_chart(render_gauge(probability), use_container_width=True)
        st.caption(
            f"{successes:,} of {SIMULATION_COUNT:,} runs cleared the readiness "
            "window under these inputs. This is a scenario result, not a certified "
            "forecast of launch success."
        )
    with right:
        st.plotly_chart(
            render_weather_history(daily_weather, era5_mode),
            use_container_width=True,
        )
        if era5_mode:
            st.caption(
                f"ERA5-derived daily screening for {calendar.month_name[int(month)]} "
                f"{year}; {len(daily_weather)} days. Green: no threshold exceeded; "
                "amber: one or more exceeded."
            )
        else:
            st.caption("Synthetic day-by-day scenario; not observed weather.")

    schedule_column, bottleneck_column = st.columns(2, gap="large")
    with schedule_column:
        st.plotly_chart(
            render_schedule(risk_percentages, mean_weather_scrub),
            use_container_width=True,
        )
        st.caption(
            "Schedule bars use expected supplier delays. Amber marks the expected "
            "critical path; the purple bar summarizes sampled weather scrubs."
        )
    with bottleneck_column:
        st.plotly_chart(render_bottlenecks(bottlenecks), use_container_width=True)

    export = pd.DataFrame(
        {
            "simulation": np.arange(1, SIMULATION_COUNT + 1),
            "supplier_slippage_days": supplier_slippage,
            "weather_scrub_days": weather_scrub_days,
            "total_slippage_days": total_slippage,
            "within_7_day_window": total_slippage < LAUNCH_WINDOW_DAYS,
        }
    )
    st.download_button(
        "Download simulation runs (CSV)",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name="asteria_launch_readiness_simulations.csv",
        mime="text/csv",
    )

    with st.expander("Evidence, assumptions and data provenance", expanded=False):
        weather_evidence = (
            "ERA5 reanalysis, retrieved through the Open-Meteo Historical Weather API"
            if era5_mode
            else "synthetic manual scenario"
        )
        screening_description = (
            "ERA5-derived daily records exceeded at least one current screening threshold"
            if era5_mode
            else "synthetic daily flags were generated for this risk frequency"
        )
        weather_limitation = (
            "These are historical point data, not a launch-day forecast."
            if era5_mode
            else "These values are synthetic and are not observed weather."
        )
        requested_latitude = float(site["latitude"])
        requested_longitude = float(site["longitude"])
        grid_latitude = grid["latitude"]
        grid_longitude = grid["longitude"]
        st.markdown(
            f"""
            **Weather evidence:** {weather_evidence}.
            The requested point ({requested_latitude:.3f}, {requested_longitude:.3f})
            resolves to grid point ({grid_latitude:.3f}, {grid_longitude:.3f});
            the selected month contains {len(daily_weather)} daily values.

            **{weather_rate_label} screening rate:** {reference_weather_risk}% of days;
            {screening_description}. **Scenario weather risk:**
            {scenario_weather_risk}%. {weather_limitation} Daily screening thresholds
            are editable proxies, not official launch commit criteria.

            **Supplier evidence:** all five subsystem names, baseline risk
            probabilities and delay-impact distributions are illustrative synthetic
            inputs. Replace them with customer-authorized, appropriately protected
            schedule data before presenting any result as customer-specific.

            **Model scope:** each supplier risk is sampled independently, delay
            impact uses a triangular distribution, and the dependency graph is a
            simplified project network. Weather blocks are bootstrapped from the
            selected month and scenario-adjusted to the selected risk percentage.
            The selected seasonal pattern is held constant when supplier slippage
            moves readiness. The output is the chance of modeled readiness within
            seven days—not the probability that a launch or mission succeeds.

            **Prototype sources:** [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
            · [Copernicus ERA5 dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels).
            Check data-provider terms and attribution requirements before any
            commercial deployment; this prototype does not imply ESA endorsement.
            """
        )

    with st.expander("Incubation-stage product hypothesis"):
        st.markdown(
            """
            **Problem hypothesis — not yet customer-validated:** space-project
            teams may reconcile supplier milestones and launch-site weather in
            separate tools, making schedule trade-offs hard to compare.

            **Proposed value:** a transparent scenario workspace that combines
            customer-entered supply-chain assumptions with a space-site weather
            context, highlights schedule bottlenecks, and makes uncertainty visible.

            **Validation milestone:** interview mission and launch-program
            planners; test whether this is a frequent, costly decision; and seek
            permission to evaluate the model on one anonymized schedule. No customer,
            pilot, validated savings, or supplier dataset is claimed in this demo.
            """
        )

    st.divider()
    st.caption(
        "Asteria is an independent concept prototype. It is not an operational "
        "launch system, flight-safety tool, or ESA-endorsed product."
    )


if __name__ == "__main__":
    main()
