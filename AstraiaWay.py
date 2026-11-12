"""Launch Supply Chain & Risk Simulator MVP.

Run with: streamlit run app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


SIMULATION_COUNT = 1_000
LAUNCH_WINDOW_DAYS = 7
BASELINE_SEED = 2026
SIMULATION_SEED = 71_903
BASELINE_WEATHER_RISK_PCT = int(
    np.random.default_rng(BASELINE_SEED + 1).integers(10, 41)
)


@dataclass(frozen=True)
class Subsystem:
    name: str
    supplier: str
    baseline_risk_pct: int
    duration_days: int
    delay_low_days: float
    delay_mode_days: float
    delay_high_days: float
    dependencies: Tuple[str, ...]


def generate_subsystems() -> List[Subsystem]:
    """Create stable, randomized mock supplier and schedule data."""
    rng = np.random.default_rng(BASELINE_SEED)
    definitions = [
        ("1st Stage Engine", 25, 2.0, 6.0, 14.0, ()),
        ("Avionics", 18, 1.0, 4.0, 10.0, ()),
        ("Ground Support Equipment", 14, 2.0, 5.0, 12.0, ()),
        ("Propellant", 10, 1.0, 3.0, 8.0, ("Ground Support Equipment",)),
        ("Payload Integration", 12, 2.0, 5.0, 11.0, ("1st Stage Engine", "Avionics")),
    ]
    suppliers = [
        "Apex Propulsion Systems",
        "Northstar Avionics",
        "Frontier Launch Services",
        "Helios Propellant Logistics",
        "Pioneer Payload Works",
    ]
    risks = rng.integers(8, 36, size=len(definitions))

    return [
        Subsystem(
            name=name,
            supplier=suppliers[index],
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


def schedule_project(
    durations: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """Calculate earliest task starts/finishes using the subsystem dependency graph."""
    starts: Dict[str, float] = {}
    finishes: Dict[str, float] = {}
    for subsystem in SUBSYSTEMS:
        start = max((finishes[name] for name in subsystem.dependencies), default=0.0)
        starts[subsystem.name] = start
        finishes[subsystem.name] = start + durations[subsystem.name]

    terminal_tasks = ("Propellant", "Payload Integration")
    project_finish = max(finishes[name] for name in terminal_tasks)
    return starts, finishes, project_finish


@st.cache_data(show_spinner=False)
def run_simulation(
    risk_percentages: Tuple[int, ...],
    weather_risk_pct: int,
) -> Tuple[float, pd.DataFrame, np.ndarray]:
    """Run a reproducible, vectorized Monte Carlo launch-window assessment."""
    seed_offset = weather_risk_pct + sum(
        (index + 1) * risk for index, risk in enumerate(risk_percentages)
    )
    rng = np.random.default_rng(SIMULATION_SEED + seed_offset)
    subsystem_delays: Dict[str, np.ndarray] = {}
    subsystem_finishes: Dict[str, np.ndarray] = {}

    for subsystem, risk_pct in zip(SUBSYSTEMS, risk_percentages):
        delay_occurs = rng.random(SIMULATION_COUNT) < risk_pct / 100.0
        impact_days = rng.triangular(
            subsystem.delay_low_days,
            subsystem.delay_mode_days,
            subsystem.delay_high_days,
            size=SIMULATION_COUNT,
        )
        delays = delay_occurs * impact_days
        subsystem_delays[subsystem.name] = delays
        dependency_finish = np.zeros(SIMULATION_COUNT)
        if subsystem.dependencies:
            dependency_finish = np.maximum.reduce(
                [subsystem_finishes[name] for name in subsystem.dependencies]
            )
        subsystem_finishes[subsystem.name] = (
            dependency_finish + subsystem.duration_days + delays
        )

    baseline_durations = {
        subsystem.name: float(subsystem.duration_days) for subsystem in SUBSYSTEMS
    }
    _, _, baseline_finish = schedule_project(baseline_durations)
    project_finish = np.maximum(
        subsystem_finishes["Propellant"],
        subsystem_finishes["Payload Integration"],
    )

    weather_occurs = rng.random(SIMULATION_COUNT) < weather_risk_pct / 100.0
    weather_impact_days = rng.triangular(1.0, 2.0, 4.0, size=SIMULATION_COUNT)
    weather_delay = weather_occurs * weather_impact_days
    total_slippage = (project_finish - baseline_finish) + weather_delay
    success_probability = float(
        np.mean(total_slippage <= LAUNCH_WINDOW_DAYS) * 100.0
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
    return success_probability, bottlenecks, total_slippage


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


def render_gauge(success_probability: float) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=success_probability,
            number={"suffix": "%", "font": {"size": 46}},
            title={"text": "Launch Success Probability"},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "#1677B8"},
                "steps": [
                    {"range": [0, 60], "color": "#FDE2E2"},
                    {"range": [60, 85], "color": "#FFF0C2"},
                    {"range": [85, 100], "color": "#DDF3E4"},
                ],
                "threshold": {
                    "line": {"color": "#172B4D", "width": 4},
                    "thickness": 0.8,
                    "value": 85,
                },
            },
        )
    )
    figure.update_layout(height=290, margin={"l": 25, "r": 25, "t": 65, "b": 15})
    return figure


def render_gantt(risk_percentages: Tuple[int, ...], weather_risk_pct: int) -> go.Figure:
    """Show the expected schedule, marking the critical path and launch window."""
    expected_durations = {
        subsystem.name: subsystem.duration_days
        + risk_pct
        / 100.0
        * (subsystem.delay_low_days + subsystem.delay_mode_days + subsystem.delay_high_days)
        / 3.0
        for subsystem, risk_pct in zip(SUBSYSTEMS, risk_percentages)
    }
    starts, finishes, expected_finish = schedule_project(expected_durations)
    critical_path = find_critical_path(starts, finishes)
    labels = [subsystem.name for subsystem in SUBSYSTEMS]
    starts_in_days = [starts[name] for name in labels]
    durations = [expected_durations[name] for name in labels]
    colors = [
        "#D97706" if name in critical_path else "#1677B8" for name in labels
    ]
    expected_weather_scrub = weather_risk_pct / 100.0 * (1.0 + 2.0 + 4.0) / 3.0
    labels.append("Weather / Launch Attempt")
    starts_in_days.append(expected_finish)
    durations.append(expected_weather_scrub)
    colors.append("#7357A5")

    figure = go.Figure(
        go.Bar(
            x=durations,
            y=labels,
            base=starts_in_days,
            orientation="h",
            marker={"color": colors},
            text=[f"{duration:.1f} d" for duration in durations],
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
        x0=expected_finish,
        x1=expected_finish + LAUNCH_WINDOW_DAYS,
        fillcolor="#DDF3E4",
        opacity=0.5,
        line_width=0,
        annotation_text="7-day launch window",
        annotation_position="top",
    )
    figure.add_vline(
        x=expected_finish,
        line_dash="dash",
        line_color="#287A3E",
        annotation_text="Expected readiness",
        annotation_position="bottom",
    )
    figure.update_layout(
        title="Expected Subsystem Schedule",
        xaxis_title="Days from project start",
        yaxis_title="",
        yaxis={"autorange": "reversed"},
        xaxis={"rangemode": "tozero", "showgrid": True, "gridcolor": "#E8EDF2"},
        height=340,
        margin={"l": 20, "r": 25, "t": 75, "b": 45},
        plot_bgcolor="white",
    )
    return figure


def render_bottlenecks(bottlenecks: pd.DataFrame) -> go.Figure:
    chart_data = bottlenecks.sort_values("Average Delay (days)", ascending=True)
    colors = [
        "#D97706" if index == len(chart_data) - 1 else "#73A9CF"
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
            hovertemplate="%{y}<br>Average delay: %{x:.2f} days<extra></extra>",
        )
    )
    figure.update_layout(
        title="Average Supplier Delay Across Simulations",
        xaxis_title="Average delay (days)",
        yaxis_title="",
        height=340,
        margin={"l": 20, "r": 35, "t": 65, "b": 45},
        plot_bgcolor="white",
    )
    return figure


def main() -> None:
    st.set_page_config(
        page_title="Launch Supply Chain & Risk Simulator",
        page_icon="🚀",
        layout="wide",
    )
    st.title("Launch Supply Chain & Risk Simulator")
    st.caption(
        "B2G launch-readiness planning using a 1,000-run Monte Carlo model. "
        "Mock supplier data is illustrative and not operational guidance."
    )

    st.sidebar.header("Scenario Controls")
    st.sidebar.caption(
        f"Launch window: {LAUNCH_WINDOW_DAYS} days after baseline readiness"
    )
    weather_risk_pct = st.sidebar.slider(
        "Weather risk (%)",
        min_value=0,
        max_value=100,
        value=BASELINE_WEATHER_RISK_PCT,
        help="Chance of weather delaying the launch attempt within the window.",
    )
    st.sidebar.subheader("Supplier / Subsystem Delay Risk")
    selected_risks: List[int] = []
    for subsystem in SUBSYSTEMS:
        selected_risks.append(
            st.sidebar.slider(
                f"{subsystem.name} ({subsystem.supplier})",
                min_value=0,
                max_value=100,
                value=subsystem.baseline_risk_pct,
                help=(
                    f"Baseline schedule: {subsystem.duration_days} days. "
                    "Risk is the chance of a supplier delay event."
                ),
                key=f"risk_{subsystem.name}",
            )
        )
    risk_percentages = tuple(selected_risks)

    probability, bottlenecks, slippage = run_simulation(
        risk_percentages, weather_risk_pct
    )
    median_slippage = float(np.median(slippage))
    p90_slippage = float(np.percentile(slippage, 90))
    st.caption(
        f"Baseline launch readiness: day 37 · "
        f"Scenario median slippage: {median_slippage:.1f} days · "
        f"90th percentile: {p90_slippage:.1f} days"
    )

    gauge_column, metric_column = st.columns([1.2, 1])
    with gauge_column:
        st.plotly_chart(render_gauge(probability), use_container_width=True)
    with metric_column:
        st.subheader("Simulation Summary")
        st.metric("Launch window success", f"{probability:.1f}%")
        st.metric("Launch window", f"{LAUNCH_WINDOW_DAYS} days")
        st.metric("Monte Carlo runs", f"{SIMULATION_COUNT:,}")
        st.info(
            "A successful run reaches subsystem readiness and clears any simulated "
            "weather scrub within the 7-day window."
        )

    schedule_column, bottleneck_column = st.columns(2)
    with schedule_column:
        st.plotly_chart(
            render_gantt(risk_percentages, weather_risk_pct),
            use_container_width=True,
        )
        st.caption("Orange bars indicate the expected critical path.")
    with bottleneck_column:
        st.plotly_chart(render_bottlenecks(bottlenecks), use_container_width=True)

    with st.expander("Baseline subsystem and supplier assumptions"):
        assumptions = pd.DataFrame(
            [
                {
                    "Subsystem": subsystem.name,
                    "Supplier (mock)": subsystem.supplier,
                    "Baseline delay risk": f"{subsystem.baseline_risk_pct}%",
                    "Scenario delay risk": f"{risk_pct}%",
                    "Baseline duration (days)": subsystem.duration_days,
                    "Delay impact (days, triangular)": (
                        f"{subsystem.delay_low_days:g} / "
                        f"{subsystem.delay_mode_days:g} / "
                        f"{subsystem.delay_high_days:g}"
                    ),
                    "Dependencies": ", ".join(subsystem.dependencies) or "None",
                }
                for subsystem, risk_pct in zip(SUBSYSTEMS, risk_percentages)
            ]
        )
        st.dataframe(assumptions, use_container_width=True, hide_index=True)
        st.caption(
            f"Mock baseline weather risk: {BASELINE_WEATHER_RISK_PCT}% "
            f"(current scenario: {weather_risk_pct}%)."
        )

    with st.expander("Model notes"):
        st.markdown(
            "- Each subsystem has a Bernoulli supplier-delay event; event duration "
            "is sampled from that subsystem's triangular impact range.\n"
            "- Dependencies are scheduled with a forward pass, so parallel work "
            "does not add unnecessary duration to the critical path.\n"
            "- Weather independently causes a 1–4 day launch scrub when its risk "
            "event occurs.\n"
            "- Success means total schedule slippage plus weather scrub is no more "
            f"than {LAUNCH_WINDOW_DAYS} days. Mock assumptions should be replaced "
            "with validated program data before operational use."
        )


if __name__ == "__main__":
    main()
