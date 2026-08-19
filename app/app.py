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


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Environment"),
        ui.p("Parameters are applied together because the spatial eigendecomposition is the expensive step.", class_="sidebar-note"),
        ui.input_numeric("temporal_length_scale", "Temporal length scale lₜ [min]", value=60.0, min=0.001, step=1.0),
        ui.input_numeric("temporal_sigma", "Temporal kernel std. σₜ", value=2.0, min=0.001, step=0.1),
        ui.input_numeric("spatial_length_scale", "Spatial length scale lₛ [km]", value=2.0, min=0.001, step=0.1),
        ui.input_numeric("spatial_sigma", "Spatial kernel std. σₛ", value=1.0, min=0.001, step=0.1),
        ui.hr(),
        ui.input_numeric("domain_size", "Square-domain side length L [km]", value=5.0, min=0.25, step=0.25),
        ui.input_select("grid_spacing", "Grid spacing δ [km]", choices={"1.0": "1.0", "0.5": "0.5", "0.25": "0.25"}, selected="0.5"),
        ui.input_action_button("apply_environment", "Apply environment", class_="btn-primary w-100"),
        ui.output_ui("environment_status"),
        width=350,
        open="desktop",
    ),
    ui.head_content(
        ui.tags.meta(name="description", content="Interactive continuous-time clarity lower-bound calculator for randomized sensor networks."),
        ui.tags.script(src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js", async_="true"),
        ui.tags.style(
            """
            :root { --accent: #2f6fdd; --ink: #172033; --muted: #657085; --panel: #ffffff; }
            body { color: var(--ink); background: #f5f7fb; }
            .bslib-page-title { font-weight: 650; }
            .hero { padding: 0.25rem 0 1rem; }
            .hero h1 { font-size: clamp(1.7rem, 3vw, 2.45rem); font-weight: 700; margin-bottom: 0.3rem; }
            .hero p { color: var(--muted); max-width: 820px; margin-bottom: 0; }
            .card { border: 1px solid #e2e7f0; box-shadow: 0 5px 18px rgba(25, 42, 72, 0.055); }
            .card-header { background: var(--panel); font-weight: 650; border-bottom-color: #e8ecf3; }
            .metric-label { color: var(--muted); font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.035em; }
            .metric-value { font-size: 1.72rem; line-height: 1.15; font-weight: 700; margin-top: 0.15rem; }
            .metric-detail { color: var(--muted); font-size: 0.9rem; margin-top: 0.3rem; }
            .result-banner { border-radius: 0.65rem; padding: 0.8rem 0.95rem; margin-top: 0.8rem; }
            .result-pass { background: #e8f7ee; color: #17633a; border: 1px solid #b9e5c9; }
            .result-fail { background: #fff1f0; color: #8e2a23; border: 1px solid #f0c6c2; }
            .sidebar-note, .assumption-list, .finite-grid-note { color: var(--muted); font-size: 0.9rem; }
            .pending { color: #8a5c00; margin-top: 0.6rem; font-size: 0.9rem; }
            .applied { color: #246642; margin-top: 0.6rem; font-size: 0.9rem; }
            .shiny-plot-output { min-height: 330px; }
            @media (max-width: 768px) { .hero h1 { font-size: 1.65rem; } .metric-value { font-size: 1.45rem; } }
            """
        ),
    ),
    ui.div(
        ui.div(
            ui.h1("Kalman–Bucy Filtering with Randomized Sensing"),
            ui.p("A browser-based design tool for the finite-grid continuous-time lower bound on expected spatially averaged clarity."),
            class_="hero",
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Sensing configuration"),
                ui.input_numeric("sensor_count", "Number of sensors Nᵣ", value=7, min=1, max=500, step=1),
                ui.input_numeric("sigma_m_squared", "Measurement noise variance σm²", value=10.0, min=0.0001, step=0.5),
                ui.input_numeric("sampling_interval", "Sampling interval Δt [min]", value=0.02, min=0.0001, step=0.01),
                ui.input_slider("target_clarity", "Target clarity q_target", min=0.1, max=0.95, value=0.7, step=0.05),
                ui.hr(),
                ui.div(ui.div("Sensing parameter θ", class_="metric-label"), ui.output_text("theta_value"), class_="metric-value"),
                ui.div(ui.output_text("noise_values"), class_="metric-detail"),
                ui.hr(),
                ui.div(ui.div("Steady-state clarity lower bound", class_="metric-label"), ui.output_text("clarity_value"), class_="metric-value"),
                ui.div(r"\(\bar q_{\Delta^\Pi_\infty}\), finite-grid continuous-time result", class_="metric-detail"),
                ui.output_ui("target_status"),
            ),
            ui.card(
                ui.card_header("Interactive Sensor-Count Design Curve"),
                ui.output_plot("clarity_plot", height="390px"),
                ui.p("The curve is the analytical lower bound only; it is not the Monte Carlo estimate shown in the paper's Fig. 4(a).", class_="finite-grid-note"),
            ),
            col_widths=(4, 8),
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Minimum sensors required"),
                ui.output_ui("minimum_sensors"),
            ),
            ui.card(
                ui.card_header("Theoretical basis and assumptions"),
                ui.p(r"Theorem 7 and Eq. (17) give the closed-form Riccati solution. Theorem 20 characterizes the grid-refinement limit of \(\bar q_{\Delta^\Pi_\infty}\)."),
                ui.p(r"This app computes its finite-grid counterpart with \(\theta=N_r/\sigma_c^2=N_r/(\sigma_m^2\Delta t)\)."),
                ui.tags.ul(
                    ui.tags.li("Matérn-1/2 spatial and temporal kernels with B₀ = 1 and q_c = 1."),
                    ui.tags.li("Sensors are sampled independently and uniformly from the grid at each sensing time."),
                    ui.tags.li("The spatial domain is square and sensing locations lie on the grid."),
                    ui.tags.li("The result is a continuous-time steady-state clarity lower bound, not a discrete-filter realization."),
                    class_="assumption-list",
                ),
            ),
            col_widths=(4, 8),
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
        return ui.div(f"Applied grid: N_g = {environment.spectrum.grid_points}", class_="applied")

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

    @render.text
    def noise_values() -> str:
        _, sigma_c_squared, _, _ = current_inputs()
        sigma_m_squared = float(input.sigma_m_squared())
        return (
            f"σm = {np.sqrt(sigma_m_squared):.4g}; "
            f"derived σc² = σm²Δt = {sigma_c_squared:.4g}"
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
                ui.strong("Continuous-time bound meets the target"),
                ui.div(f"{clarity:.3f} ≥ {target:.3f}"),
                class_="result-banner result-pass",
            )
        return ui.div(
            ui.strong("Continuous-time bound is below the target"),
            ui.div(f"{clarity:.3f} < {target:.3f}"),
            class_="result-banner result-fail",
        )

    @render.ui
    def minimum_sensors():
        result = optimum()
        if result is None:
            return ui.div(
                ui.h3(f"> {MAXIMUM_SENSORS}"),
                ui.p("The analytical lower bound does not reach the selected target within the search range."),
            )
        sensor_count, clarity = result
        return ui.div(
            ui.h3(str(sensor_count)),
            ui.p(f"First Nᵣ with bound ≥ target; bound = {clarity:.4f}."),
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
