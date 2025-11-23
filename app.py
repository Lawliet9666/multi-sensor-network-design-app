import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import eigh

# ==========================================
# 1. Mathematical Implementation based on Paper
# ==========================================

def compute_spatial_eigenvalues(domain_size, resolution, sigma_s, l_s):
    """
    Computes the eigenvalues (nu_i) of the Spatial Kernel Matrix K_gg.
    Based on Section 5.1 and Lemma 18.
    """
    # Create Grid
    N = int(domain_size/resolution) +1
    x = np.linspace(0, domain_size, N)
    y = np.linspace(0, domain_size, N)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
    N_g = grid_points.shape[0]
    # D = N_g * resolution * resolution  # Domain area
    D =  domain_size * domain_size  # Domain area 

    # Compute Distance Matrix
    dists = cdist(grid_points, grid_points, metric='euclidean')

    # Spatial Kernel (Matern-1/2 / Exponential) - Eq. 33
    K_gg = (sigma_s**2) * np.exp(-dists / l_s)

    # Solve Eigenvalues
    # We use eigh because K_gg is symmetric
    eigvals = eigh(K_gg/ N_g * D, eigvals_only=True)
    
    # Sort descending
    nu = eigvals[::-1]
    
    # Normalize eigenvalues for the limit definition (Theorem 19 proof)
    # nu_i approximation
    # nu = eigvals / N_g 
    nu_D = nu / D 
    # 
    return nu_D, N_g

def calculate_steady_state_clarity(N_r, sigma_m, delta_t, l_t, sigma_t, nu_D_vals):
    """
    Calculates the steady-state lower bound of averaged expected clarity.
    Implements Theorem 19 (Eq. 74) and Theorem 8 (Eq. 27).
    """
    # 1. Sensing Parameter theta (Eq. 75/67)
    theta = N_r / ((sigma_m**2) * delta_t)

    # 2. System Dynamics Parameters (Section 5.4)
    # A = aI, Qc = qcI
    # A0 = -1/lt, so a = -1/lt
    # B0 = 1, so Qc = B0*B0.T = 1 (assuming scalar temporal process per point)
    a = -1.0 / l_t
    q_c = 1.0 
    
    # C0 derived from Section 5.4 state space rep
    C0 = sigma_t * np.sqrt(2.0 / l_t)
    C0_sq = C0**2

    # 3. Calculate Eigenvalues of averaged information matrix G_bar (Eq. 77 context)
    # The term inside Gamma in Theorem 19 is (theta * C0^2 * nu_i)
    lambda_G = theta * C0_sq * nu_D_vals

    # 4. Calculate Gamma_i (Eq. 27)
    # gamma_i = -qc / (a - sqrt(a^2 + qc * lambda_G_i))
    # Note: a is negative.
    gamma_vals = -q_c / (a - np.sqrt(a**2 + q_c * lambda_G))

    # 5. Calculate Final Clarity Limit (Eq. 74)
    # q = 1 / (1 + C0^2 * sum(nu_i * gamma_i))
    summation_term = np.sum(nu_D_vals * gamma_vals)
    q_infinity = 1.0 / (1.0 + C0_sq * summation_term)

    return q_infinity

# ==========================================
# 2. Streamlit UI Layout
# ==========================================

st.set_page_config(page_title="STGPKF Sensor Design Tool", layout="wide")

st.title("Kalman-Bucy Filtering with Randomized Sensing")
st.subheader("Fundamental Limits & Sensor Network Design")

st.markdown("""
This tool implements the theoretical framework from **"Kalman-Bucy Filtering with Randomized Sensing: Fundamental Limits and Sensor Network Design for Field Estimation
", Xinyi Wang, Devansh R. Agrawal, Dimitra Panagou **.
It calculates the **Steady-State Lower Bound of Averaged Expected Clarity** ($q_{\Delta_\Pi^\infty}$) to help design sensor networks.
""")

# --- Sidebar: Parameters ---
st.sidebar.header("1. Environment Parameters")
st.sidebar.markdown("*Spatiotemporal GP Kernel settings*")
l_t = st.sidebar.number_input("Temporal Length Scale ($l_t$)", value=60.0)
sigma_t = st.sidebar.number_input("Temporal Kernel Std ($\sigma_t$)", value=2.0)
l_s = st.sidebar.number_input("Spatial Length Scale ($l_s$)", value=2.0)
sigma_s = st.sidebar.number_input("Spatial Kernel Std ($\sigma_s$)", value=1.0)

st.sidebar.header("2. Grid Settings")
st.sidebar.markdown("*Approximation for Integral Operator*")
dom_size = st.sidebar.number_input("Domain Size ($km$)", value=5.0)
res = st.sidebar.select_slider("Grid Resolution (Running speed trade-off)", options=[1.0, 0.5, 0.25], value=0.5)

st.sidebar.header("3. Design Constraints")
q_target = st.sidebar.slider("Target Clarity ($q_{target}$)", 0.0, 1.0, 0.70)

# --- Main Computation Cache ---
# We cache the eigenvalue computation because it only changes if environment/grid changes
@st.cache_data
def get_cached_eigenvalues(d_size, r, s_s, len_s):
    return compute_spatial_eigenvalues(d_size, r, s_s, len_s)

with st.spinner("Computing Spatial Kernel Eigenvalues..."):
    nu_vals, N_g = get_cached_eigenvalues(dom_size, res, sigma_s, l_s)

st.sidebar.success(f"Grid Points ($N_g$): {N_g}")

# --- Interactive Sensing Design ---

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Sensing Configuration")
    st.info("Adjust these to see if you meet the target.")
    
    N_r = st.number_input("Number of Sensors ($N_r$)", min_value=1, max_value=100, value=7, step=1)
    sigma_m = st.number_input("Measurement Noise ($\sigma_m$)", value=2.0)
    delta_t = st.number_input("Sampling Interval ($\Delta t$)", value=0.05)

    # Calculate current clarity
    current_q = calculate_steady_state_clarity(N_r, sigma_m, delta_t, l_t, sigma_t, nu_vals)
    
    # Calculate Theta
    theta = N_r / ((sigma_m**2) * delta_t)
    
    st.markdown("---")
    st.markdown(fr"**Sensing Parameter ($\theta$):** {theta:.2f}")
    st.markdown(fr"**Achieved Clarity ($q_{{\Delta^\Pi_\infty}}$):** {current_q:.4f}")
    
    if current_q >= q_target:
        st.success(f"✅ Target Met! ({current_q:.3f} >= {q_target})")
    else:
        st.error(f"❌ Below Target! Need {q_target - current_q:.3f} more.")

with col2:
    st.markdown("### Design Analysis")
    
    tab1, tab2 = st.tabs(["Sensors vs Clarity", "Optimization"])
    
    with tab1:
        st.markdown("**Impact of Number of Sensors (Replicating Fig. 5)**")
        
        # Generate curve data
        nr_range = np.arange(1, 101, 2)
        q_curve = []
        for n in nr_range:
            q = calculate_steady_state_clarity(n, sigma_m, delta_t, l_t, sigma_t, nu_vals)
            q_curve.append(q)
            
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(nr_range, q_curve, label='Steady-State Bound of Clarity', color='blue', marker='.')
        ax.axhline(y=q_target, color='r', linestyle='--', label=f'Target Clarity ({q_target})')
        ax.axvline(x=N_r, color='g', linestyle=':', label=fr'Current Number of Sensors ($N_r$) ({N_r})')
        ax.set_xlabel("Number of Sensors ($N_r$)")
        ax.set_ylabel("Clarity")
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()
        st.pyplot(fig)
        
    with tab2:
        st.markdown("**Minimum Sensors Required**")
        st.write("Automatically calculating the minimum $N_r$ required to satisfy $q_{target}$ given current noise and sampling rate.")
        
        if st.button("Calculate Optimal N_r"):
            found = False
            for n_opt in range(1, 500):
                q_check = calculate_steady_state_clarity(n_opt, sigma_m, delta_t, l_t, sigma_t, nu_vals)
                if q_check >= q_target:
                    st.metric(label="Minimum Sensors Required", value=n_opt, delta=f"Clarity: {q_check:.3f}")
                    found = True
                    break
            if not found:
                st.warning("Could not satisfy target within 500 sensors.")

# --- Theory Section ---
st.markdown("---")
st.markdown("### Theoretical Basis")
st.latex(r"""
\lim_{\delta \to 0} \bar{q}_{\Delta^\Pi_\infty} =
\frac{1}{\;1 + C_0^{2} \displaystyle\sum_{i=1}^{\infty} \frac{\nu_i}{|D|}\, \gamma\!\left( \theta C_0^{2} \frac{\nu_i}{|D|} \right)}
""")
st.markdown("""
Where:
* $|D|$ is the spatial domain area.
* $\\nu_i$ are eigenvalues of the spatial kernel integral operator.
* $\\theta = \\frac{N_r}{\sigma_m^2 \Delta t}$ is the sensing parameter.
* $\gamma(\cdot)$ associated with the closedform Riccati solution in Theorem 2.
""")

