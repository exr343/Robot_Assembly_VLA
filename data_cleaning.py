#!/usr/bin/env python3
import argparse
import glob
import os
import pickle

import numpy as np
from tqdm import tqdm


def _to_np(x):
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _xyz_from_step(step):
    state = step.get("observation", {}).get("state")
    if state is None:
        return None
    arr = _to_np(state)
    if arr.shape[0] < 3:
        return None
    return arr[:3]


def clean_steps(steps, tol):
    n = len(steps)
    cleaned = []

    # --- Filter 1: drop near-duplicate consecutive states using full state vector ---
    if n >= 1:
        cleaned.append(steps[0])
        prev_state = steps[0].get("observation", {}).get("state")
    else:
        prev_state = None

    for step in steps[1:-1]:
        cur_state = step.get("observation", {}).get("state")

        if prev_state is None or cur_state is None:
            cleaned.append(step)
            prev_state = cur_state
            continue

        diff = _to_np(cur_state) - _to_np(prev_state)
        if np.linalg.norm(diff) >= tol:
            cleaned.append(step)
            prev_state = cur_state

    if n > 1:
        cleaned.append(steps[-1])

    # --- Filter 2: drop steps with small XYZ displacement relative to mean displacement ---
    m = len(cleaned)
    if m <= 2:
        return cleaned  # already includes first/last policy

    dists = []
    for i in range(m - 1):
        a = _xyz_from_step(cleaned[i])
        b = _xyz_from_step(cleaned[i + 1])
        if a is None or b is None:
            continue
        dists.append(np.linalg.norm(b - a))

    if not dists:
        return cleaned

    thresh = 0.5 * float(np.mean(dists))

    filtered = [cleaned[0]]
    prev_xyz = _xyz_from_step(cleaned[0])

    for step in cleaned[1:-1]:
        cur_xyz = _xyz_from_step(step)

        if prev_xyz is None or cur_xyz is None:
            filtered.append(step)
            prev_xyz = cur_xyz
            continue

        if np.linalg.norm(cur_xyz - prev_xyz) >= thresh:
            filtered.append(step)
            prev_xyz = cur_xyz

    filtered.append(cleaned[-1])  # always keep last
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Remove consecutive steps with near-identical state, then remove small XYZ displacements."
    )
    parser.add_argument(
        "--in-dir",
        default="/scratch/pioneer/users/exr343/RLDS_step_front_side_wrist",
        help="Directory containing episode pickle files.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for cleaned pickles. Defaults to <in-dir>_cleaned.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="L2 norm threshold for consecutive state similarity (Filter 1).",
    )
    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir or f"{in_dir}_cleaned"
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(in_dir, "*.pkl")))
    if not files:
        raise SystemExit(f"No pickle files found in {in_dir}")

    pbar = tqdm(files, unit="file")
    for fp in pbar:
        with open(fp, "rb") as f:
            data = pickle.load(f)

        steps = data.get("steps", [])
        n0 = len(steps)
        new_steps = clean_steps(steps, args.tol)
        n1 = len(new_steps)
        removed_pct = ((n0 - n1) / n0 * 100.0) if n0 else 0.0

        data["steps"] = new_steps

        pbar.set_description(os.path.basename(fp))
        pbar.set_postfix_str(f"removed={removed_pct:.1f}%")

        out_fp = os.path.join(out_dir, os.path.basename(fp))
        with open(out_fp, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
