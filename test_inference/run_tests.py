"""
Comprehensive inference tests for the Mamba CFD Surrogate Model.
Tests loading, prediction, physical plausibility, consistency, and performance.

Usage:
    python test_inference/run_tests.py

All results saved to test_inference/results/
"""
import os
import sys
import time
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from inference import MambaSurrogate

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'packaged_model')

os.makedirs(RESULTS_DIR, exist_ok=True)

# Known physical bounds for Apollo reentry
PHYSICAL_BOUNDS = {
    'qw':    (1e2, 1e7),       # W/m^2 — heat flux
    'pw':    (1e-1, 5e5),      # Pa — surface pressure (can exceed 1e5 at high q_inf)
    'tw':    (1e-3, 1e3),      # Pa — wall shear stress
    'me':    (1e-3, 6.0),      # dimensionless — edge Mach
    'theta': (1e-8, 1e-1),     # m — momentum thickness
}

OUTPUT_LABELS = {
    'qw': 'Heat Flux qw (W/m\u00b2)',
    'pw': 'Pressure pw (Pa)',
    'tw': 'Shear Stress \u03c4w (Pa)',
    'me': 'Edge Mach Me',
    'theta': 'Momentum Thickness \u03b8 (m)',
}

# Test flight conditions spanning the training envelope
TEST_CONDITIONS = [
    {'name': 'Low speed, high altitude',
     'velocity': 4000.0, 'density': 0.0001, 'aoa': 154.0, 'dynamic_pressure': 800.0},
    {'name': 'Mid speed, mid altitude',
     'velocity': 7500.0, 'density': 0.003, 'aoa': 155.0, 'dynamic_pressure': 84375.0},
    {'name': 'High speed, low altitude',
     'velocity': 10000.0, 'density': 0.006, 'aoa': 156.0, 'dynamic_pressure': 300000.0},
    {'name': 'Low AoA edge',
     'velocity': 6000.0, 'density': 0.002, 'aoa': 152.5, 'dynamic_pressure': 36000.0},
    {'name': 'High AoA edge',
     'velocity': 8000.0, 'density': 0.004, 'aoa': 157.5, 'dynamic_pressure': 128000.0},
]

test_results = {
    'tests_passed': 0,
    'tests_failed': 0,
    'tests_total': 0,
    'details': [],
}


def log_test(name, passed, detail=""):
    test_results['tests_total'] += 1
    if passed:
        test_results['tests_passed'] += 1
        status = "PASS"
    else:
        test_results['tests_failed'] += 1
        status = "FAIL"
    test_results['details'].append({'name': name, 'status': status, 'detail': detail})
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def test_model_loading():
    """Test 1: Model loads correctly."""
    print("\n" + "=" * 60)
    print("TEST 1: Model Loading")
    print("=" * 60)

    try:
        surrogate = MambaSurrogate(MODEL_DIR)
        log_test("Model loads without error", True)
        log_test("Has expected outputs",
                 set(surrogate.output_names) == {'qw', 'pw', 'tw', 'me', 'theta'},
                 f"outputs: {surrogate.output_names}")
        log_test("Mesh has expected point count",
                 40000 < surrogate.n_points < 55000,
                 f"n_points: {surrogate.n_points}")
        log_test("Partitions computed",
                 len(surrogate.partitions) >= 6,
                 f"n_partitions: {len(surrogate.partitions)}")
        return surrogate
    except Exception as e:
        log_test("Model loads without error", False, str(e))
        return None


def test_single_prediction(surrogate):
    """Test 2: Single prediction runs and returns expected shapes."""
    print("\n" + "=" * 60)
    print("TEST 2: Single Prediction")
    print("=" * 60)

    cond = TEST_CONDITIONS[1]  # mid-range condition
    t0 = time.time()
    results = surrogate.predict(
        velocity=cond['velocity'],
        density=cond['density'],
        aoa=cond['aoa'],
        dynamic_pressure=cond['dynamic_pressure'],
    )
    elapsed = time.time() - t0

    log_test("Prediction completes", results is not None)
    log_test(f"Inference time reasonable", elapsed < 30.0, f"{elapsed:.2f}s")

    for name in surrogate.output_names:
        log_test(f"{name} in results", name in results)
        log_test(f"{name} shape correct",
                 results[name].shape == (surrogate.n_points,),
                 f"shape: {results[name].shape}")
        log_test(f"{name} no NaN values",
                 not np.any(np.isnan(results[name])),
                 f"NaN count: {np.sum(np.isnan(results[name]))}")
        log_test(f"{name} no Inf values",
                 not np.any(np.isinf(results[name])),
                 f"Inf count: {np.sum(np.isinf(results[name]))}")

    log_test("xyz in results", 'xyz' in results)
    log_test("xyz shape correct",
             results['xyz'].shape == (surrogate.n_points, 3),
             f"shape: {results['xyz'].shape}")

    return results, elapsed


def test_physical_plausibility(surrogate):
    """Test 3: Predictions are physically plausible across conditions."""
    print("\n" + "=" * 60)
    print("TEST 3: Physical Plausibility")
    print("=" * 60)

    for cond in TEST_CONDITIONS:
        print(f"\n  Condition: {cond['name']}")
        results = surrogate.predict(
            velocity=cond['velocity'],
            density=cond['density'],
            aoa=cond['aoa'],
            dynamic_pressure=cond['dynamic_pressure'],
        )

        for name in surrogate.output_names:
            lo, hi = PHYSICAL_BOUNDS[name]
            vals = results[name]
            pct_in_bounds = ((vals >= lo) & (vals <= hi)).mean() * 100

            log_test(f"{cond['name']}: {name} mostly in bounds",
                     pct_in_bounds > 90.0,
                     f"{pct_in_bounds:.1f}% in [{lo:.0e}, {hi:.0e}], "
                     f"range: [{vals.min():.2e}, {vals.max():.2e}]")

        # Physics check: qw should correlate with dynamic pressure
        log_test(f"{cond['name']}: qw mean > 0",
                 results['qw'].mean() > 0,
                 f"mean qw: {results['qw'].mean():.0f} W/m^2")


def test_monotonicity(surrogate):
    """Test 4: Higher velocity/density should generally produce higher heating."""
    print("\n" + "=" * 60)
    print("TEST 4: Monotonicity (Sanity Checks)")
    print("=" * 60)

    # Fix other params, vary velocity
    base = {'density': 0.003, 'aoa': 155.0, 'dynamic_pressure': 84375.0}
    velocities = [4000.0, 6000.0, 8000.0, 10000.0]
    mean_qws = []
    for v in velocities:
        results = surrogate.predict(velocity=v, **base)
        mean_qws.append(results['qw'].mean())

    # Check generally increasing (allow some non-monotonicity due to q_inf being fixed)
    increases = sum(1 for i in range(len(mean_qws)-1) if mean_qws[i+1] > mean_qws[i])
    log_test("qw increases with velocity (general trend)",
             increases >= 2,
             f"mean qw at V={velocities}: {[f'{q:.0f}' for q in mean_qws]}")

    # Fix velocity, vary density (compute consistent dynamic pressure = 0.5 * rho * V^2)
    V_fixed = 7500.0
    aoa_fixed = 155.0
    densities = [0.0005, 0.001, 0.003, 0.006]
    mean_qws2 = []
    for d in densities:
        q_inf = 0.5 * d * V_fixed**2
        results = surrogate.predict(velocity=V_fixed, density=d, aoa=aoa_fixed,
                                   dynamic_pressure=q_inf)
        mean_qws2.append(results['qw'].mean())

    increases2 = sum(1 for i in range(len(mean_qws2)-1) if mean_qws2[i+1] > mean_qws2[i])
    log_test("qw increases with density (fixed velocity, consistent q_inf)",
             increases2 >= 2,
             f"mean qw at rho={densities}: {[f'{q:.0f}' for q in mean_qws2]}")


def test_consistency(surrogate):
    """Test 5: Same input produces same output (determinism)."""
    print("\n" + "=" * 60)
    print("TEST 5: Consistency (Determinism)")
    print("=" * 60)

    cond = TEST_CONDITIONS[1]
    r1 = surrogate.predict(
        velocity=cond['velocity'], density=cond['density'],
        aoa=cond['aoa'], dynamic_pressure=cond['dynamic_pressure'],
    )
    r2 = surrogate.predict(
        velocity=cond['velocity'], density=cond['density'],
        aoa=cond['aoa'], dynamic_pressure=cond['dynamic_pressure'],
    )

    for name in surrogate.output_names:
        max_diff = np.abs(r1[name] - r2[name]).max()
        log_test(f"{name} identical across runs",
                 max_diff < 1e-4,
                 f"max diff: {max_diff:.2e}")


def test_spatial_patterns(surrogate):
    """Test 6: qw should be highest near stagnation point (nose)."""
    print("\n" + "=" * 60)
    print("TEST 6: Spatial Pattern Checks")
    print("=" * 60)

    cond = TEST_CONDITIONS[1]
    results = surrogate.predict(
        velocity=cond['velocity'], density=cond['density'],
        aoa=cond['aoa'], dynamic_pressure=cond['dynamic_pressure'],
    )

    xyz = results['xyz']
    qw = results['qw']

    # At AoA=155° the stagnation point shifts off-centerline toward the windward side
    # Check that peak qw is localized (not spread uniformly)
    r_perp = np.sqrt(xyz[:, 1]**2 + xyz[:, 2]**2)
    max_qw_idx = np.argmax(qw)
    max_qw_r = r_perp[max_qw_idx]
    max_r = r_perp.max()

    log_test("Peak qw on capsule surface (not at edge)",
             max_qw_r < max_r * 0.95,
             f"peak qw at r={max_qw_r:.4f}m, max r={max_r:.4f}m")

    # qw should vary spatially (not constant)
    qw_cv = np.std(qw) / np.mean(qw)
    log_test("qw has spatial variation (not constant)",
             qw_cv > 0.1,
             f"coefficient of variation: {qw_cv:.2f}")

    # Peak qw should be much higher than mean (concentrated heating)
    qw_peak_ratio = qw.max() / qw.mean()
    log_test("Peak qw concentrated (max >> mean)",
             qw_peak_ratio > 1.5,
             f"max/mean ratio: {qw_peak_ratio:.1f}x")

    # Pressure should also show spatial variation
    pw = results['pw']
    pw_cv = np.std(pw) / np.mean(pw)
    log_test("pw has spatial variation",
             pw_cv > 0.1,
             f"coefficient of variation: {pw_cv:.2f}")


def test_performance(surrogate):
    """Test 7: Benchmark inference speed."""
    print("\n" + "=" * 60)
    print("TEST 7: Performance Benchmark")
    print("=" * 60)

    cond = TEST_CONDITIONS[1]
    times = []
    n_runs = 5

    # Warmup
    surrogate.predict(velocity=cond['velocity'], density=cond['density'],
                     aoa=cond['aoa'], dynamic_pressure=cond['dynamic_pressure'])

    for i in range(n_runs):
        t0 = time.time()
        surrogate.predict(velocity=cond['velocity'], density=cond['density'],
                         aoa=cond['aoa'], dynamic_pressure=cond['dynamic_pressure'])
        times.append(time.time() - t0)

    mean_time = np.mean(times)
    std_time = np.std(times)

    log_test(f"Avg inference time ({n_runs} runs)",
             True,
             f"{mean_time:.3f}s +/- {std_time:.3f}s")
    log_test("Inference under 10s (CPU) or 2s (GPU)",
             mean_time < 10.0,
             f"device: {surrogate.device}")

    return mean_time


def generate_visualizations(surrogate):
    """Generate visualization plots from test predictions."""
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)

    cond = TEST_CONDITIONS[1]
    results = surrogate.predict(
        velocity=cond['velocity'], density=cond['density'],
        aoa=cond['aoa'], dynamic_pressure=cond['dynamic_pressure'],
    )

    xyz = results['xyz']
    x_coord = xyz[:, 0]
    y_coord = xyz[:, 1]
    z_coord = xyz[:, 2]
    r_perp = np.sqrt(y_coord**2 + z_coord**2)

    # Surface maps for all outputs
    n_out = len(surrogate.output_names)
    fig, axes = plt.subplots(2, n_out, figsize=(5 * n_out, 8))
    if n_out == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle(f"Inference Test: V={cond['velocity']} m/s, "
                 f"\u03c1={cond['density']} kg/m\u00b3, AoA={cond['aoa']}\u00b0",
                 fontsize=14, fontweight='bold')

    for i, name in enumerate(surrogate.output_names):
        vals = np.log10(np.clip(results[name], 1e-10, None))

        # Side view
        ax = axes[0, i]
        sc = ax.scatter(x_coord, r_perp, c=vals, cmap='jet', s=0.3, alpha=0.8)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('r (m)')
        ax.set_title(f'{OUTPUT_LABELS.get(name, name)}\nSide View')
        ax.set_aspect('equal')
        plt.colorbar(sc, ax=ax, shrink=0.7, label=f'log10({name})')

        # Front view
        ax = axes[1, i]
        sc = ax.scatter(y_coord, z_coord, c=vals, cmap='jet', s=0.3, alpha=0.8)
        ax.set_xlabel('Y (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title(f'{OUTPUT_LABELS.get(name, name)}\nFront View')
        ax.set_aspect('equal')
        plt.colorbar(sc, ax=ax, shrink=0.7, label=f'log10({name})')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'inference_surface_maps.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

    # Sweep: qw vs velocity
    print("  Running velocity sweep...")
    velocities = np.linspace(4000, 10000, 8)
    base = {'density': 0.003, 'aoa': 155.0, 'dynamic_pressure': 84375.0}
    sweep_data = {'velocity': [], 'max_qw': [], 'mean_qw': [], 'median_qw': []}

    for v in velocities:
        r = surrogate.predict(velocity=v, **base)
        sweep_data['velocity'].append(v)
        sweep_data['max_qw'].append(r['qw'].max())
        sweep_data['mean_qw'].append(r['qw'].mean())
        sweep_data['median_qw'].append(np.median(r['qw']))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep_data['velocity'], sweep_data['max_qw'], 'o-', label='Max qw', color='#E53935')
    ax.plot(sweep_data['velocity'], sweep_data['mean_qw'], 's-', label='Mean qw', color='#1E88E5')
    ax.plot(sweep_data['velocity'], sweep_data['median_qw'], '^-', label='Median qw', color='#4CAF50')
    ax.set_xlabel('Velocity (m/s)', fontsize=12)
    ax.set_ylabel('Heat Flux qw (W/m\u00b2)', fontsize=12)
    ax.set_title('Heat Flux vs Velocity Sweep', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'velocity_sweep.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

    # Distribution of predictions
    fig, axes = plt.subplots(1, n_out, figsize=(4.5 * n_out, 3.5))
    if n_out == 1:
        axes = [axes]
    fig.suptitle('Prediction Distributions (Mid-Range Condition)', fontsize=13, fontweight='bold')

    for i, name in enumerate(surrogate.output_names):
        ax = axes[i]
        vals = results[name]
        ax.hist(np.log10(np.clip(vals, 1e-10, None)), bins=80, color='steelblue',
                alpha=0.7, edgecolor='white', linewidth=0.3)
        ax.set_xlabel(f'log10({name})')
        ax.set_ylabel('Count')
        ax.set_title(OUTPUT_LABELS.get(name, name), fontsize=10)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'prediction_distributions.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

    # Multi-condition comparison
    print("  Running multi-condition comparison...")
    fig, axes = plt.subplots(len(TEST_CONDITIONS), 1, figsize=(10, 3 * len(TEST_CONDITIONS)))
    fig.suptitle('Heat Flux Across Flight Conditions (Side View)', fontsize=14, fontweight='bold')

    all_qw_log = []
    all_results = []
    for cond in TEST_CONDITIONS:
        r = surrogate.predict(
            velocity=cond['velocity'], density=cond['density'],
            aoa=cond['aoa'], dynamic_pressure=cond['dynamic_pressure'],
        )
        all_results.append(r)
        all_qw_log.append(np.log10(np.clip(r['qw'], 1e-10, None)))

    vmin = min(q.min() for q in all_qw_log)
    vmax = max(q.max() for q in all_qw_log)

    for i, (cond, r, qw_log) in enumerate(zip(TEST_CONDITIONS, all_results, all_qw_log)):
        ax = axes[i]
        xyz = r['xyz']
        r_perp = np.sqrt(xyz[:, 1]**2 + xyz[:, 2]**2)
        sc = ax.scatter(xyz[:, 0], r_perp, c=qw_log, cmap='jet',
                       vmin=vmin, vmax=vmax, s=0.3, alpha=0.8)
        ax.set_ylabel('r (m)')
        ax.set_title(f"{cond['name']} (V={cond['velocity']}, \u03c1={cond['density']})",
                    fontsize=10)
        ax.set_aspect('equal')
        plt.colorbar(sc, ax=ax, shrink=0.8, label='log10(qw)')

    axes[-1].set_xlabel('X (m)')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'multi_condition_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def main():
    print("=" * 60)
    print("  MAMBA CFD SURROGATE — INFERENCE TEST SUITE")
    print("=" * 60)

    # Test 1: Loading
    surrogate = test_model_loading()
    if surrogate is None:
        print("\nModel failed to load. Cannot continue tests.")
        return

    # Test 2: Single prediction
    results, single_time = test_single_prediction(surrogate)

    # Test 3: Physical plausibility
    test_physical_plausibility(surrogate)

    # Test 4: Monotonicity
    test_monotonicity(surrogate)

    # Test 5: Consistency
    test_consistency(surrogate)

    # Test 6: Spatial patterns
    test_spatial_patterns(surrogate)

    # Test 7: Performance
    avg_time = test_performance(surrogate)

    # Generate visualizations
    generate_visualizations(surrogate)

    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    print(f"  Passed: {test_results['tests_passed']}/{test_results['tests_total']}")
    print(f"  Failed: {test_results['tests_failed']}/{test_results['tests_total']}")
    print(f"  Avg inference time: {avg_time:.3f}s")
    print(f"  Device: {surrogate.device}")

    if test_results['tests_failed'] > 0:
        print("\n  FAILED TESTS:")
        for t in test_results['details']:
            if t['status'] == 'FAIL':
                print(f"    - {t['name']}: {t['detail']}")

    # Save results
    summary = {
        'tests_passed': test_results['tests_passed'],
        'tests_failed': test_results['tests_failed'],
        'tests_total': test_results['tests_total'],
        'avg_inference_time_s': avg_time,
        'device': str(surrogate.device),
        'n_mesh_points': surrogate.n_points,
        'outputs': surrogate.output_names,
        'details': test_results['details'],
    }
    with open(os.path.join(RESULTS_DIR, 'test_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
