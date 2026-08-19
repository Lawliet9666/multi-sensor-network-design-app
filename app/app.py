"""Browser-ready Shiny interface for sensor-network clarity design."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from shiny import App, Inputs, Outputs, Session, reactive, render, ui

from model import (
    SpatialSpectrum,
    compute_spatial_spectrum,
    continuous_measurement_noise_intensity,
    minimum_sensor_count,
    sensing_parameter,
    steady_state_clarity_lower_bound,
)


MAXIMUM_SENSORS = 500
DEFAULT_DOMAIN_SIZE = 5.0
DEFAULT_GRID_SPACING = 0.5
DEFAULT_TEMPORAL_LENGTH_SCALE = 60.0
DEFAULT_TEMPORAL_SIGMA = 2.0
DEFAULT_SPATIAL_LENGTH_SCALE = 2.0
DEFAULT_SPATIAL_SIGMA = 1.0


@dataclass(frozen=True)
class AppliedEnvironment:
    spectrum: SpatialSpectrum
    domain_size: float
    temporal_length_scale: float
    temporal_sigma: float
    spatial_length_scale: float
    spatial_sigma: float


def build_environment(
    domain_size: float,
    grid_spacing: float,
    temporal_length_scale: float,
    temporal_sigma: float,
    spatial_length_scale: float,
    spatial_sigma: float,
) -> AppliedEnvironment:
    spectrum = compute_spatial_spectrum(
        domain_size,
        grid_spacing,
        spatial_sigma,
        spatial_length_scale,
    )
    if temporal_length_scale <= 0.0 or temporal_sigma <= 0.0:
        raise ValueError("Temporal kernel parameters must be greater than zero.")
    return AppliedEnvironment(
        spectrum=spectrum,
        domain_size=float(domain_size),
        temporal_length_scale=float(temporal_length_scale),
        temporal_sigma=float(temporal_sigma),
        spatial_length_scale=float(spatial_length_scale),
        spatial_sigma=float(spatial_sigma),
    )


def parameter_label(text: str, symbol: str, unit: str | None = None):
    """Build a consistent text, LaTeX-symbol, and optional-unit label."""

    suffix = f" [{unit}]" if unit is not None else ""
    return ui.span(text, " ", ui.HTML(rf"\(({symbol})\)"), suffix)


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("1. Environment Parameters"),
        ui.input_numeric(
            "temporal_length_scale",
            parameter_label("Temporal Length Scale", r"\ell_t", "min"),
            value=60.0,
            min=0.001,
            step=1.0,
        ),
        ui.input_numeric(
            "temporal_sigma",
            parameter_label("Temporal Kernel Std", r"\sigma_t"),
            value=2.0,
            min=0.001,
            step=0.1,
        ),
        ui.input_numeric(
            "spatial_length_scale",
            parameter_label("Spatial Length Scale", r"\ell_s", "km"),
            value=2.0,
            min=0.001,
            step=0.1,
        ),
        ui.input_numeric(
            "spatial_sigma",
            parameter_label("Spatial Kernel Std", r"\sigma_s"),
            value=1.0,
            min=0.001,
            step=0.1,
        ),
        ui.hr(),
        ui.h4("2. Grid Settings"),
        ui.input_numeric(
            "domain_size",
            parameter_label("Square-Domain Side Length", "L", "km"),
            value=5.0,
            min=0.25,
            step=0.25,
        ),
        ui.input_select(
            "grid_spacing",
            parameter_label("Grid Spacing", r"\delta", "km"),
            choices={"1.0": "1.0", "0.5": "0.5", "0.25": "0.25"},
            selected="0.5",
        ),
        ui.input_action_button("apply_environment", "Apply environment", class_="btn-primary w-100"),
        ui.output_ui("environment_status"),
        width=350,
        open="desktop",
    ),
    ui.head_content(
        ui.tags.meta(name="description", content="Interactive continuous-time clarity lower-bound calculator for randomized sensor networks."),
        ui.tags.script(src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js", async_="true"),
        ui.tags.script(
            """
            window.addEventListener("load", () => {
                window.jQuery?.(document).on("shiny:value", (event) => {
                    window.setTimeout(() => {
                        if (window.MathJax?.typesetPromise) {
                            window.MathJax.typesetPromise([event.target]);
                        }
                    }, 0);
                });
            });
            """
        ),
        ui.tags.style(
            """
            :root { --accent: #2f6fdd; --ink: #172033; --muted: #657085; --panel: #ffffff; }
            body { color: var(--ink); background: #f5f7fb; }
            .bslib-page-title { font-weight: 650; }
            .hero { padding: 0.25rem 0 1rem; }
            .hero h1 { font-size: clamp(1.7rem, 3vw, 2.45rem); font-weight: 700; margin-bottom: 0.3rem; }
            .hero h2 { font-size: 1.2rem; font-weight: 650; margin: 0 0 0.35rem; }
            .hero p { color: var(--muted); margin-bottom: 0; }
            .card { border: 1px solid #e2e7f0; box-shadow: 0 5px 18px rgba(25, 42, 72, 0.055); }
            .card-header { background: var(--panel); font-weight: 650; border-bottom-color: #e8ecf3; }
            .metric-row { display: flex; align-items: baseline; flex-wrap: nowrap; gap: 0.25rem; margin-top: 0.9rem; font-size: 0.95rem; line-height: 1.5; white-space: nowrap; }
            .metric-label { color: var(--ink); font-size: 0.95rem; font-weight: 650; }
            .metric-inline-value { font-variant-numeric: tabular-nums; }
            .metric-detail { color: var(--muted); font-size: 0.86rem; margin-top: 0.15rem; white-space: nowrap; }
            .result-banner { display: flex; align-items: center; gap: 0.55rem; border-radius: 0.65rem; padding: 0.85rem 1rem; margin-top: 0.95rem; font-size: 0.95rem; }
            .result-icon { display: inline-grid; flex: 0 0 auto; place-items: center; width: 1.35rem; height: 1.35rem; border-radius: 0.25rem; color: #fff; font-weight: 700; line-height: 1; }
            .result-pass .result-icon { background: #16a34a; }
            .result-fail .result-icon { background: #c2413b; }
            .result-pass { background: #eaf7ed; color: #17633a; border: 0; }
            .result-fail { background: #fff1f0; color: #8e2a23; border: 0; }
            .sensor-design-equation { font-size: 1rem; overflow-x: auto; }
            .sensor-design-equation mjx-container[display="true"] { margin: 0.55rem 0 !important; }
            .design-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); grid-template-areas: "configuration curve" "minimum curve"; grid-template-rows: auto minmax(0, auto); gap: 1rem; align-items: stretch; margin-bottom: 1rem; }
            .design-grid > .card { margin: 0; }
            .sensing-card { grid-area: configuration; }
            .minimum-card { grid-area: minimum; }
            .curve-card { grid-area: curve; min-height: 0; }
            .curve-card .shiny-plot-output { flex: 1 1 auto; min-height: 430px; height: 100% !important; }
            .theory-card { margin-bottom: 0; }
            .theory-grid { display: grid; grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr); gap: 1rem; }
            .theory-block { background: #f8fafc; border: 1px solid #e6ebf2; border-radius: 0.7rem; padding: 1rem 1.1rem; }
            .theory-heading { color: var(--ink); font-size: 1rem; font-weight: 650; margin: 0 0 0.75rem; }
            .assumption-list { color: var(--muted); font-size: 0.9rem; line-height: 1.55; margin: 0; padding-left: 1.2rem; }
            .assumption-list li + li { margin-top: 0.35rem; }
            .conclusion-copy { color: #354056; font-size: 0.9rem; line-height: 1.55; }
            .conclusion-copy p { margin-bottom: 0.65rem; }
            .sidebar-note { color: var(--muted); font-size: 0.9rem; }
            .pending { color: #8a5c00; margin-top: 0.6rem; font-size: 0.9rem; }
            .applied { color: #246642; margin-top: 0.6rem; font-size: 0.9rem; }
            .shiny-plot-output { min-height: 330px; }
            @media (min-width: 1500px) { .hero-purpose { white-space: nowrap; } }
            @media (max-width: 991px) {
                .design-grid { grid-template-columns: 1fr; grid-template-areas: "configuration" "minimum" "curve"; grid-template-rows: auto; }
                .curve-card .shiny-plot-output { min-height: 360px; height: 360px !important; }
                .theory-grid { grid-template-columns: 1fr; }
            }
            @media (max-width: 768px) { .hero h1 { font-size: 1.65rem; } }
            """
        ),
    ),
    ui.div(
        ui.div(
            ui.h1("Kalman–Bucy Filtering with Randomized Sensing"),
            ui.h2("Fundamental Limits & Sensor Network Design"),
            ui.p(
                "This tool implements the theoretical framework from ",
                ui.strong(
                    '"Kalman-Bucy Filtering with Randomized Sensing: Fundamental Limits and Sensor Network Design for Field Estimation", '
                    "Xinyi Wang, Devansh R. Agrawal, Dimitra Panagou"
                ),
                ui.span(
                    ". It calculates the ",
                    ui.strong("Steady-State Lower Bound of Averaged Expected Clarity"),
                    r" \(\bar q_{\Delta^\Pi_\infty}\) to help design sensor networks.",
                    class_="hero-purpose",
                ),
            ),
            class_="hero",
        ),
        ui.div(
            ui.card(
                ui.card_header("Sensing configuration"),
                ui.input_numeric(
                    "sensor_count",
                    parameter_label("Number of Sensors", r"N_r"),
                    value=7,
                    min=1,
                    max=500,
                    step=1,
                ),
                ui.input_numeric(
                    "sigma_m_squared",
                    parameter_label("Measurement Noise Var", r"\sigma_m^2"),
                    value=10.0,
                    min=0.0001,
                    step=0.5,
                ),
                ui.input_numeric(
                    "sampling_interval",
                    parameter_label("Sampling Interval", r"\Delta t", "min"),
                    value=0.02,
                    min=0.0001,
                    step=0.01,
                ),
                ui.input_slider(
                    "target_clarity",
                    parameter_label("Target Clarity", r"q_{\mathrm{target}}"),
                    min=0.1,
                    max=0.95,
                    value=0.7,
                    step=0.05,
                ),
                ui.hr(),
                ui.div(
                    ui.span(
                        parameter_label("Sensing Parameter", r"\theta"),
                        ":",
                        class_="metric-label",
                    ),
                    ui.span(
                        ui.output_text("theta_value", inline=True),
                        class_="metric-inline-value",
                    ),
                    class_="metric-row",
                ),
                ui.div(ui.output_ui("noise_values"), class_="metric-detail"),
                ui.div(
                    ui.span(
                        parameter_label(
                            "Steady-State Clarity Lower Bound",
                            r"\bar q_{\Delta^\Pi_\infty}",
                        ),
                        ":",
                        class_="metric-label",
                    ),
                    ui.span(
                        ui.output_text("clarity_value", inline=True),
                        class_="metric-inline-value",
                    ),
                    class_="metric-row",
                ),
                ui.output_ui("target_status"),
                class_="sensing-card",
            ),
            ui.card(
                ui.card_header("Minimum sensors required"),
                ui.output_ui("minimum_sensors"),
                class_="minimum-card",
            ),
            ui.card(
                ui.card_header("Interactive Sensor-Count Design Curve"),
                ui.output_plot("clarity_plot", height="100%", fill=True),
                class_="curve-card",
            ),
            class_="design-grid",
        ),
        ui.card(
            ui.card_header("Assumptions and conclusion"),
            ui.div(
                ui.tags.section(
                    ui.h5("Assumptions", class_="theory-heading"),
                    ui.tags.ul(
                        ui.tags.li(
                            "Matérn-1/2 spatial and temporal kernels, with ",
                            ui.HTML(r"\(B_0=1\)"),
                            " and ",
                            ui.HTML(r"\(q_c=1\)"),
                            ".",
                        ),
                        ui.tags.li(
                            "The spatial domain is square and discretized by a uniform grid; "
                            "all sensing locations lie on the grid."
                        ),
                        ui.tags.li(
                            "At each sensing time, ",
                            ui.HTML(r"\(N_r\)"),
                            " sensor locations are sampled independently and uniformly from the grid.",
                        ),
                        ui.tags.li(
                            "The measurement model uses the continuous-time noise intensity ",
                            ui.HTML(r"\(\sigma_c^2=\sigma_m^2\Delta t\)"),
                            ".",
                        ),
                        class_="assumption-list",
                    ),
                    class_="theory-block",
                ),
                ui.tags.section(
                    ui.h5("Conclusion", class_="theory-heading"),
                    ui.div(
                        ui.p(
                            "Under these assumptions, Theorem 7 and Eq. (17) give the closed-form "
                            "continuous-time steady-state covariance upper bound. Theorem 15 then gives "
                            "the finite-grid lower bound on expected spatially averaged clarity ",
                            ui.HTML(r"\(\bar q_{\Delta^\Pi_\infty}\)"),
                            ".",
                        ),
                        ui.p(
                            "Theorem 20 shows that, with the environment fixed, this clarity lower "
                            "bound converges to a finite grid-independent limit as ",
                            ui.HTML(r"\(\delta\to0\)"),
                            ". The limit is governed by ",
                            ui.HTML(r"\(\theta=N_r/\sigma_c^2=N_r/(\sigma_m^2\Delta t)\)"),
                            ".",
                        ),
                        class_="conclusion-copy",
                    ),
                    class_="theory-block",
                ),
                class_="theory-grid",
            ),
            class_="theory-card",
        ),
        class_="container-fluid px-3 px-lg-4 pb-4",
    ),
    title="STGPKF Sensor Design",
    fillable=True,
)


def server(input: Inputs, output: Outputs, session: Session) -> None:
    applied_environment = reactive.value(
        build_environment(
            DEFAULT_DOMAIN_SIZE,
            DEFAULT_GRID_SPACING,
            DEFAULT_TEMPORAL_LENGTH_SCALE,
            DEFAULT_TEMPORAL_SIGMA,
            DEFAULT_SPATIAL_LENGTH_SCALE,
            DEFAULT_SPATIAL_SIGMA,
        )
    )

    def requested_environment() -> tuple[float, float, float, float, float, float]:
        return (
            float(input.domain_size()),
            float(input.grid_spacing()),
            float(input.temporal_length_scale()),
            float(input.temporal_sigma()),
            float(input.spatial_length_scale()),
            float(input.spatial_sigma()),
        )

    @reactive.effect
    @reactive.event(input.apply_environment)
    def apply_environment() -> None:
        try:
            values = requested_environment()
            applied_environment.set(build_environment(*values))
            ui.notification_show("Environment applied.", type="message", duration=3)
        except (ValueError, FloatingPointError) as error:
            ui.notification_show(str(error), type="error", duration=7)

    @render.ui
    def environment_status():
        environment = applied_environment.get()
        requested = requested_environment()
        applied = (
            environment.domain_size,
            environment.spectrum.spacing,
            environment.temporal_length_scale,
            environment.temporal_sigma,
            environment.spatial_length_scale,
            environment.spatial_sigma,
        )
        if not np.allclose(requested, applied, rtol=0.0, atol=1e-12):
            return ui.div("Changes pending — click Apply environment.", class_="pending")
        return ui.div(
            "Applied Grid: ",
            ui.HTML(r"\(N_g\)"),
            f" = {environment.spectrum.grid_points}",
            class_="applied",
        )

    @reactive.calc
    def current_inputs() -> tuple[int, float, float, float]:
        sensor_count_value = float(input.sensor_count())
        sigma_m_squared = float(input.sigma_m_squared())
        sampling_interval = float(input.sampling_interval())
        target = float(input.target_clarity())
        sigma_c_squared = continuous_measurement_noise_intensity(
            sigma_m_squared, sampling_interval
        )
        sensing_parameter(sensor_count_value, sigma_c_squared)
        if not 0.0 < target < 1.0:
            raise ValueError("Target clarity must lie strictly between zero and one.")
        return int(sensor_count_value), sigma_c_squared, sampling_interval, target

    @reactive.calc
    def current_clarity() -> float:
        sensor_count, sigma_c_squared, _, _ = current_inputs()
        environment = applied_environment.get()
        return steady_state_clarity_lower_bound(
            sensor_count,
            sigma_c_squared,
            environment.temporal_length_scale,
            environment.temporal_sigma,
            environment.spectrum.eigenvalues,
        )

    @reactive.calc
    def optimum() -> tuple[int, float] | None:
        _, sigma_c_squared, _, target = current_inputs()
        environment = applied_environment.get()
        return minimum_sensor_count(
            target,
            MAXIMUM_SENSORS,
            sigma_c_squared,
            environment.temporal_length_scale,
            environment.temporal_sigma,
            environment.spectrum.eigenvalues,
        )

    @render.text
    def theta_value() -> str:
        sensor_count, sigma_c_squared, _, _ = current_inputs()
        return f"{sensing_parameter(sensor_count, sigma_c_squared):.3f}"

    @render.ui
    def noise_values():
        _, sigma_c_squared, _, _ = current_inputs()
        sigma_m_squared = float(input.sigma_m_squared())
        return ui.span(
            ui.HTML(
                rf"\(\sigma_m={np.sqrt(sigma_m_squared):.4g},\quad "
                rf"\sigma_c^2={sigma_c_squared:.4g}\)"
            )
        )

    @render.text
    def clarity_value() -> str:
        return f"{current_clarity():.4f}"

    @render.ui
    def target_status():
        _, _, _, target = current_inputs()
        clarity = current_clarity()
        if clarity >= target:
            return ui.div(
                ui.span("✓", class_="result-icon", aria_hidden="true"),
                ui.span(ui.strong("Target Met!"), f" ({clarity:.3f} ≥ {target:.3f})"),
                class_="result-banner result-pass",
            )
        return ui.div(
            ui.span("×", class_="result-icon", aria_hidden="true"),
            ui.span(ui.strong("Target Not Met"), f" ({clarity:.3f} < {target:.3f})"),
            class_="result-banner result-fail",
        )

    @render.ui
    def minimum_sensors():
        result = optimum()
        design_criterion = ui.div(
            ui.HTML(
                r"\[\min\;N_r\quad\text{s.t.}\quad"
                r"\bar q_{\Delta^\Pi_\infty}\ge q_{\mathrm{target}}.\]"
            ),
            class_="sensor-design-equation",
        )
        if result is None:
            return ui.div(
                ui.h3(f"> {MAXIMUM_SENSORS}"),
                design_criterion,
                ui.p("The analytical lower bound does not reach the selected target within the search range."),
            )
        sensor_count, _ = result
        return ui.div(
            ui.h3(str(sensor_count)),
            design_criterion,
        )

    @render.plot
    def clarity_plot():
        sensor_count, sigma_c_squared, _, target = current_inputs()
        environment = applied_environment.get()
        result = optimum()
        optimum_count = result[0] if result is not None else 0
        plot_maximum = min(MAXIMUM_SENSORS, max(120, sensor_count, optimum_count))
        sensor_counts = np.arange(1, plot_maximum + 1)
        clarity_values = np.array(
            [
                steady_state_clarity_lower_bound(
                    int(count),
                    sigma_c_squared,
                    environment.temporal_length_scale,
                    environment.temporal_sigma,
                    environment.spectrum.eigenvalues,
                )
                for count in sensor_counts
            ]
        )

        with plt.rc_context(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans"],
                "mathtext.fontset": "dejavusans",
                "font.size": 10.0,
            }
        ):
            fig, axis = plt.subplots(figsize=(6.0, 3.0))
            marker_spacing = max(1, len(sensor_counts) // 50)
            axis.plot(
                sensor_counts,
                clarity_values,
                color="blue",
                marker=".",
                markevery=marker_spacing,
                label=r"$\bar q_{\Delta^\Pi_\infty}$",
            )
            axis.axhline(
                target,
                color="red",
                linestyle="--",
                label=r"$q_{\mathrm{target}}$",
            )
            axis.axvline(
                sensor_count,
                color="green",
                linestyle=":",
                label=rf"Current $N_r={sensor_count}$",
            )
            axis.set_xlim(1, plot_maximum)
            axis.set_ylim(0.0, 1.0)
            axis.set_xlabel(r"Number of Sensors ($N_r$)")
            axis.set_ylabel("Clarity")
            axis.grid(True, which="both", linestyle="--", alpha=0.5)
            axis.legend()
            fig.tight_layout()
        return fig


app = App(app_ui, server)
