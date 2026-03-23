"""
Interactive 3D visualization of the Mamba CFD Surrogate Model.
Allows users to input flight conditions and view predicted surface quantities
as interactive 3D heatmaps on the Apollo capsule mesh.

Usage:
    streamlit run app.py
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import torch
import time
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import MambaSurrogate

# Page config
st.set_page_config(
    page_title="Apollo Reentry CFD Surrogate",
    page_icon="🚀",
    layout="wide",
)

OUTPUT_LABELS = {
    'qw': 'Heat Flux qw (W/m²)',
    'pw': 'Pressure pw (Pa)',
    'tw': 'Shear Stress τw (Pa)',
    'me': 'Edge Mach Me (-)',
    'theta': 'Momentum Thickness θ (m)',
}

OUTPUT_UNITS = {
    'qw': 'W/m²',
    'pw': 'Pa',
    'tw': 'Pa',
    'me': '',
    'theta': 'm',
}


@st.cache_resource
def load_model():
    """Load the surrogate model (cached across reruns)."""
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'packaged_model')
    return MambaSurrogate(model_dir)


def create_3d_plot(xyz, values, output_name, point_size=1.5):
    """Create an interactive 3D scatter plot of the capsule surface."""
    log_vals = np.log10(np.clip(values, 1e-10, None))

    fig = go.Figure(data=[
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=log_vals,
                colorscale='Jet',
                colorbar=dict(
                    title=dict(text=f'log10({output_name})', font=dict(color='white')),
                    tickfont=dict(color='white'),
                ),
                cmin=np.percentile(log_vals, 1),
                cmax=np.percentile(log_vals, 99),
            ),
            customdata=np.stack([values, log_vals], axis=-1),
            hovertemplate=(
                f'<b>{OUTPUT_LABELS.get(output_name, output_name)}</b><br>'
                f'Value: %{{customdata[0]:.4g}} {OUTPUT_UNITS.get(output_name, "")}<br>'
                f'log10: %{{customdata[1]:.3f}}<br>'
                'X: %{x:.4f} m<br>'
                'Y: %{y:.4f} m<br>'
                'Z: %{z:.4f} m<br>'
                '<extra></extra>'
            ),
        )
    ])

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (m)', backgroundcolor='#1a1a2e', gridcolor='#333',
                      color='white'),
            yaxis=dict(title='Y (m)', backgroundcolor='#1a1a2e', gridcolor='#333',
                      color='white'),
            zaxis=dict(title='Z (m)', backgroundcolor='#1a1a2e', gridcolor='#333',
                      color='white'),
            bgcolor='#1a1a2e',
            aspectmode='data',
        ),
        paper_bgcolor='#0B1D3A',
        plot_bgcolor='#0B1D3A',
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
    )

    return fig


def create_side_view(xyz, values, output_name):
    """Create a 2D side view (X vs r) scatter plot."""
    r_perp = np.sqrt(xyz[:, 1]**2 + xyz[:, 2]**2)
    log_vals = np.log10(np.clip(values, 1e-10, None))

    fig = go.Figure(data=[
        go.Scatter(
            x=xyz[:, 0],
            y=r_perp,
            mode='markers',
            marker=dict(
                size=1.5,
                color=log_vals,
                colorscale='Jet',
                colorbar=dict(
                    title=dict(text=f'log10({output_name})', font=dict(color='white')),
                    tickfont=dict(color='white'),
                ),
                cmin=np.percentile(log_vals, 1),
                cmax=np.percentile(log_vals, 99),
            ),
            customdata=np.stack([values, log_vals], axis=-1),
            hovertemplate=(
                f'<b>{OUTPUT_LABELS.get(output_name, output_name)}</b><br>'
                f'Value: %{{customdata[0]:.4g}} {OUTPUT_UNITS.get(output_name, "")}<br>'
                'X: %{x:.4f} m<br>'
                'r: %{y:.4f} m<br>'
                '<extra></extra>'
            ),
        )
    ])

    fig.update_layout(
        xaxis=dict(title='X (m)', color='white', gridcolor='#333'),
        yaxis=dict(title='r (m)', color='white', gridcolor='#333', scaleanchor='x'),
        paper_bgcolor='#0B1D3A',
        plot_bgcolor='#1a1a2e',
        font=dict(color='white'),
        margin=dict(l=60, r=20, t=30, b=60),
        height=400,
    )

    return fig


def main():
    # Hide deploy button
    st.markdown("""
    <style>
    [data-testid="stAppDeployButton"] {display: none;}
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    # Apollo Reentry CFD Surrogate Model
    **Mamba-3 SSM | 244K Parameters | 99.5% qw within ±5%**

    Enter flight conditions below to predict aerothermal surface quantities on the Apollo capsule.
    """)

    # Load model
    with st.spinner("Loading model..."):
        surrogate = load_model()

    # Sidebar — Flight Conditions
    st.sidebar.header("Flight Conditions")

    velocity = st.sidebar.slider(
        "Velocity (m/s)", 3000.0, 11000.0, 7500.0, step=100.0,
        help="Freestream velocity (training range: 3,000-11,000 m/s)"
    )
    density = st.sidebar.slider(
        "Density (kg/m³)", 0.00001, 0.009, 0.003, step=0.0001, format="%.5f",
        help="Atmospheric density (training range: 1.57e-5 to 8.21e-3)"
    )
    aoa = st.sidebar.slider(
        "Angle of Attack (°)", 152.0, 158.0, 155.0, step=0.5,
        help="Angle of attack (training range: 152-158°)"
    )

    # Auto-compute dynamic pressure
    dynamic_pressure = 0.5 * density * velocity**2
    st.sidebar.metric("Dynamic Pressure (Pa)", f"{dynamic_pressure:,.0f}")

    # Output selector — these don't trigger re-inference, just re-render
    st.sidebar.header("Display")
    selected_output = st.sidebar.selectbox(
        "Surface Quantity",
        surrogate.output_names,
        format_func=lambda x: OUTPUT_LABELS.get(x, x),
        key="output_select",
    )

    view_mode = st.sidebar.selectbox("View", ["3D Interactive", "Side View (2D)", "Both"],
                                     key="view_select")
    point_size = st.sidebar.slider("Point Size", 0.5, 5.0, 1.5, step=0.5, key="point_size")

    # Check if flight conditions changed
    current_conditions = {
        'velocity': velocity, 'density': density,
        'aoa': aoa, 'dynamic_pressure': dynamic_pressure,
    }
    needs_rerun = (
        'conditions' not in st.session_state
        or st.session_state['conditions'] != current_conditions
    )

    # Run prediction — only when conditions change or first time
    if st.sidebar.button("Predict", type="primary", use_container_width=True) or (
        needs_rerun and 'results' not in st.session_state
    ):
        with st.spinner(f"Running inference ({surrogate.device})... ~3s GPU / ~60s CPU"):
            t0 = time.time()
            results = surrogate.predict(
                velocity=velocity,
                density=density,
                aoa=aoa,
                dynamic_pressure=dynamic_pressure,
            )
            elapsed = time.time() - t0

        st.session_state['results'] = results
        st.session_state['elapsed'] = elapsed
        st.session_state['conditions'] = current_conditions

    # Display results
    if 'results' in st.session_state:
        results = st.session_state['results']
        elapsed = st.session_state['elapsed']
        conds = st.session_state['conditions']

        # Stats bar
        col1, col2, col3, col4, col5 = st.columns(5)
        vals = results[selected_output]
        col1.metric("Min", f"{vals.min():.4g}")
        col2.metric("Max", f"{vals.max():.4g}")
        col3.metric("Mean", f"{vals.mean():.4g}")
        col4.metric("Median", f"{np.median(vals):.4g}")
        col5.metric("Inference Time", f"{elapsed:.2f}s")

        xyz = results['xyz']

        # Visualization
        if view_mode in ["3D Interactive", "Both"]:
            st.plotly_chart(
                create_3d_plot(xyz, vals, selected_output, point_size),
                use_container_width=True,
            )

        if view_mode in ["Side View (2D)", "Both"]:
            st.plotly_chart(
                create_side_view(xyz, vals, selected_output),
                use_container_width=True,
            )

        # All outputs summary
        st.markdown("### All Outputs Summary")
        summary_cols = st.columns(len(surrogate.output_names))
        for i, name in enumerate(surrogate.output_names):
            v = results[name]
            with summary_cols[i]:
                st.markdown(f"**{OUTPUT_LABELS.get(name, name)}**")
                st.markdown(f"Min: {v.min():.4g}")
                st.markdown(f"Max: {v.max():.4g}")
                st.markdown(f"Mean: {v.mean():.4g}")

        # Condition info
        st.markdown("---")
        st.markdown(
            f"*V={conds['velocity']:.0f} m/s, "
            f"ρ={conds['density']:.5f} kg/m³, "
            f"AoA={conds['aoa']:.1f}°, "
            f"q∞={conds['dynamic_pressure']:,.0f} Pa | "
            f"{surrogate.n_points:,} mesh points | "
            f"Device: {surrogate.device}*"
        )

    else:
        st.info("Click **Predict** in the sidebar to run the model.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Model**: MambaAutoencoder (244K params)\n\n"
        "**Accuracy**: 99.5% qw within ±5%\n\n"
        "**Mesh**: 49,698 surface points\n\n"
        "**Architecture**: Mamba-3 SSM with RoPE + trapezoidal discretization"
    )


if __name__ == '__main__':
    main()
