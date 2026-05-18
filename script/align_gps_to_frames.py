#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import numpy as np


WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def load_keyframe_trajectory(path: Path):
    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"Invalid trajectory format in {path}")
    timestamps = data[:, 0]
    xyz = data[:, 1:4]
    return timestamps, xyz


def load_gps(path: Path):
    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"Invalid gps format in {path}")
    timestamps = data[:, 0]
    lon = data[:, 1]
    lat = data[:, 2]
    alt = data[:, 3]
    return timestamps, lon, lat, alt


def load_frames(path: Path):
    timestamps = []
    image_paths = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid frame line: {line}")
            timestamps.append(float(parts[0]))
            image_paths.append(parts[1])
    return np.asarray(timestamps, dtype=float), image_paths


def geodetic_to_ecef(lat_deg, lon_deg, alt):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt) * cos_lat * cos_lon
    y = (n + alt) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + alt) * sin_lat
    return np.column_stack((x, y, z))


def geodetic_to_enu(lat_deg, lon_deg, alt):
    ecef = geodetic_to_ecef(lat_deg, lon_deg, alt)
    lat0 = math.radians(lat_deg[0])
    lon0 = math.radians(lon_deg[0])
    origin = ecef[0]

    sin_lat0 = math.sin(lat0)
    cos_lat0 = math.cos(lat0)
    sin_lon0 = math.sin(lon0)
    cos_lon0 = math.cos(lon0)

    rot = np.array(
        [
            [-sin_lon0, cos_lon0, 0.0],
            [-sin_lat0 * cos_lon0, -sin_lat0 * sin_lon0, cos_lat0],
            [cos_lat0 * cos_lon0, cos_lat0 * sin_lon0, sin_lat0],
        ],
        dtype=float,
    )
    delta = (ecef - origin).T
    return (rot @ delta).T


def interpolate_positions(sample_ts, ref_ts, ref_xyz):
    x = np.interp(sample_ts, ref_ts, ref_xyz[:, 0])
    y = np.interp(sample_ts, ref_ts, ref_xyz[:, 1])
    z = np.interp(sample_ts, ref_ts, ref_xyz[:, 2])
    return np.column_stack((x, y, z))


def umeyama_alignment(src, dst):
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("Need at least three paired points for Umeyama alignment")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    cov = (dst_demean.T @ src_demean) / src.shape[0]
    u, d, vh = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vh) < 0.0:
        s[-1, -1] = -1.0

    r = u @ s @ vh
    src_var = np.mean(np.sum(src_demean * src_demean, axis=1))
    scale = np.trace(np.diag(d) @ s) / src_var
    t = dst_mean - scale * (r @ src_mean)
    return scale, r, t


def evaluate_offset(offset, kf_ts, kf_xyz, gps_ts, gps_xyz_enu, min_pairs):
    shifted_ts = kf_ts + offset
    valid = (shifted_ts >= gps_ts[0]) & (shifted_ts <= gps_ts[-1])
    if valid.sum() < min_pairs:
        return None

    matched_kf = kf_xyz[valid]
    matched_gps = interpolate_positions(shifted_ts[valid], gps_ts, gps_xyz_enu)

    try:
        scale, rotation, translation = umeyama_alignment(matched_gps, matched_kf)
    except (ValueError, np.linalg.LinAlgError):
        return None

    aligned_gps = (scale * (rotation @ matched_gps.T)).T + translation
    rmse = float(np.sqrt(np.mean(np.sum((aligned_gps - matched_kf) ** 2, axis=1))))

    return {
        "offset": offset,
        "rmse": rmse,
        "pairs": int(valid.sum()),
        "scale": float(scale),
    }


def search_best_offset(kf_ts, kf_xyz, gps_ts, gps_xyz_enu, min_pairs):
    min_offset = gps_ts[0] - kf_ts[-1]
    max_offset = gps_ts[-1] - kf_ts[0]
    steps = [0.5, 0.05, 0.005]
    windows = [None, 1.0, 0.1]
    best = None

    for step, window in zip(steps, windows):
        if best is None or window is None:
            start = min_offset
            end = max_offset
        else:
            start = max(min_offset, best["offset"] - window)
            end = min(max_offset, best["offset"] + window)

        candidate = start
        local_best = best
        while candidate <= end + 1e-9:
            result = evaluate_offset(candidate, kf_ts, kf_xyz, gps_ts, gps_xyz_enu, min_pairs)
            if result is not None and (local_best is None or result["rmse"] < local_best["rmse"]):
                local_best = result
            candidate += step

        best = local_best

    if best is None:
        raise RuntimeError("Failed to find a valid timestamp offset")
    return best


def write_frames_with_gps(output_path, frame_ts, image_paths, gps_ts, lon, lat, alt, offset):
    shifted_ts = frame_ts + offset
    valid = (shifted_ts >= gps_ts[0]) & (shifted_ts <= gps_ts[-1])
    if not np.all(valid):
        missing = int((~valid).sum())
        print(f"Warning: skipped {missing} frames outside GPS time range")

    interp_lon = np.interp(shifted_ts[valid], gps_ts, lon)
    interp_lat = np.interp(shifted_ts[valid], gps_ts, lat)
    interp_alt = np.interp(shifted_ts[valid], gps_ts, alt)

    with output_path.open("w", encoding="utf-8") as handle:
        for ts, path, lo, la, al in zip(
            frame_ts[valid], np.asarray(image_paths)[valid], interp_lon, interp_lat, interp_alt
        ):
            handle.write(f"{ts:.6f} {path} {lo:.6f} {la:.6f} {al:.6f}\n")

    return int(valid.sum())


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Align GPS timestamps to a visual keyframe trajectory and generate frames_with_gps.txt."
    )
    parser.add_argument("--kf-traj", required=True, help="Path to KeyFrameTrajectory.txt")
    parser.add_argument("--gps", required=True, help="Path to gps.txt")
    parser.add_argument("--frames", required=True, help="Path to frames.txt")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for generated frames_with_gps file. Default: sibling frames_with_gps_aligned.txt",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=20,
        help="Minimum overlapping KF/GPS pairs required when scoring an offset",
    )
    return parser


def main():
    args = build_argparser().parse_args()

    kf_path = Path(args.kf_traj)
    gps_path = Path(args.gps)
    frames_path = Path(args.frames)
    output_path = (
        Path(args.output)
        if args.output is not None
        else frames_path.with_name("frames_with_gps_aligned.txt")
    )

    kf_ts, kf_xyz = load_keyframe_trajectory(kf_path)
    gps_ts, lon, lat, alt = load_gps(gps_path)
    frame_ts, image_paths = load_frames(frames_path)
    gps_xyz_enu = geodetic_to_enu(lat, lon, alt)

    best = search_best_offset(kf_ts, kf_xyz, gps_ts, gps_xyz_enu, args.min_pairs)
    written = write_frames_with_gps(output_path, frame_ts, image_paths, gps_ts, lon, lat, alt, best["offset"])

    print(f"Best offset: {best['offset']:.6f} s")
    print(f"Alignment RMSE: {best['rmse']:.4f} m")
    print(f"Estimated visual scale: {best['scale']:.6f}")
    print(f"Matched KF/GPS pairs: {best['pairs']}")
    print(f"Written frames: {written}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
