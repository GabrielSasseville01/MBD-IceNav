"""Save/load full simulation state for scenario replay."""

import pickle


def save_simulation_state(
    polygons, centroids, bonds, thicknesses,
    positions_history, rotations_history, bonds_history, ship_history, submergence_history,
    ship_length, ship_beam, ship_draft, n_frames, dt,
    output_2d=None, output_3d=None, simple_mode=False,
    filepath=None,
):
    """Save full simulation state to a pickle file for replay (visualization without re-running)."""
    state = {
        'polygons': polygons,
        'centroids': centroids,
        'bonds': bonds,
        'thicknesses': thicknesses,
        'positions_history': positions_history,
        'rotations_history': rotations_history,
        'bonds_history': bonds_history,
        'ship_history': ship_history,
        'submergence_history': submergence_history,
        'ship_length': ship_length,
        'ship_beam': ship_beam,
        'ship_draft': ship_draft,
        'n_frames': n_frames,
        'dt': dt,
        'output_2d': output_2d,
        'output_3d': output_3d,
        'simple_mode': simple_mode,
    }
    with open(filepath, 'wb') as f:
        pickle.dump(state, f)
    print(f"  Saved state to {filepath}", flush=True)


def load_simulation_state(filepath):
    """Load simulation state from a pickle file. Returns dict with all fields needed for replay."""
    with open(filepath, 'rb') as f:
        state = pickle.load(f)
    print(f"  Loaded state from {filepath} ({state['n_frames']} frames)", flush=True)
    return state
