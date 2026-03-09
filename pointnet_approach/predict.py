"""
predict.py - Predict Break Surfaces with PointNet++
====================================================
Loads trained model, predicts on all fragments in data/without_gt/.

Usage:
  python3 predict.py
  python3 predict.py --checkpoint checkpoints/best_model.pt --gpu cuda:0
"""
import os
import sys
import glob
import time
import argparse
import numpy as np
import torch
import open3d as o3d

import config
from preprocess import preprocess
from model.pointnet2 import PointNet2Classifier


def predict_fragment(points, normals, model, device, num_points=4096,
                     batch_size=64):
    """
    Predict break/original for every point in a fragment.

    For each point, extract its neighborhood, run through model,
    get probability of break surface.

    Args:
        points: (N, 3) array
        normals: (N, 3) array
        model: Trained PointNet++ model
        device: torch device
        num_points: Points per neighborhood
        batch_size: Prediction batch size

    Returns:
        proba: (N,) probability of break surface per point
    """
    model.eval()
    n_total = len(points)
    proba = np.zeros(n_total, dtype=np.float32)

    # Build KDTree
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    tree = o3d.geometry.KDTreeFlann(pcd)

    # Process in batches
    n_batches = (n_total + batch_size - 1) // batch_size
    t0 = time.time()

    with torch.no_grad():
        for bi in range(n_batches):
            start = bi * batch_size
            end = min(start + batch_size, n_total)
            batch_features = []

            for idx in range(start, end):
                center = points[idx]

                # Find neighborhood
                k, nn_idx, nn_dist = tree.search_knn_vector_3d(center, num_points + 1)
                nn_idx = np.array(nn_idx[1:])

                if len(nn_idx) < num_points:
                    pad = np.random.choice(nn_idx, num_points - len(nn_idx), replace=True)
                    nn_idx = np.concatenate([nn_idx, pad])
                nn_idx = nn_idx[:num_points]

                # Center and build features
                patch_points = points[nn_idx] - center
                patch_normals = normals[nn_idx]
                features = np.concatenate([patch_points, patch_normals], axis=1)  # (N, 6)
                features = features.T.astype(np.float32)  # (6, N)
                batch_features.append(features)

            # Stack and predict
            batch_tensor = torch.from_numpy(np.stack(batch_features)).to(device)
            logits, _ = model(batch_tensor)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            proba[start:end] = probs

            if bi % 50 == 0 or bi == n_batches - 1:
                elapsed = time.time() - t0
                pct = 100 * end / n_total
                eta = (elapsed / max(end, 1)) * (n_total - end)
                print(f"    {pct:.0f}% ({end:,}/{n_total:,}) "
                      f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    return proba


def postprocess(pcd, proba, threshold=0.5):
    """
    Simple post-processing: threshold + small cluster removal.
    """
    points = np.asarray(pcd.points)
    preds = (proba >= threshold).astype(int)

    # Remove small clusters
    break_idx = np.where(preds == 1)[0]
    if len(break_idx) > 50:
        bp = o3d.geometry.PointCloud()
        bp.points = o3d.utility.Vector3dVector(points[break_idx])

        # Compute spacing
        tree = o3d.geometry.KDTreeFlann(pcd)
        dists = []
        for i in range(min(1000, len(points))):
            k, idx, dsq = tree.search_knn_vector_3d(points[i], 2)
            dists.append(np.sqrt(dsq[1]))
        spacing = float(np.median(dists))

        cls = np.array(bp.cluster_dbscan(eps=spacing * 3, min_points=10))
        from collections import Counter
        cc = Counter(cls[cls >= 0])
        small = set(c for c, cnt in cc.items() if cnt < 50)

        removed = 0
        for i, cidx in enumerate(break_idx):
            if cls[i] in small or cls[i] == -1:
                preds[cidx] = 0
                removed += 1

        if removed > 0:
            print(f"    Removed {removed:,} noise points")

    return preds


def save_result_ply(pcd, preds, output_path, labels=None):
    """Save colored result PLY."""
    n = len(preds)
    colors = np.zeros((n, 3))

    if labels is not None:
        for i in range(n):
            if labels[i] == 1 and preds[i] == 1:
                colors[i] = [0, 1, 0]          # TP Green
            elif labels[i] == 0 and preds[i] == 1:
                colors[i] = [1, 1, 0]          # FP Yellow
            elif labels[i] == 1 and preds[i] == 0:
                colors[i] = [1, 0, 0]          # FN Red
            else:
                colors[i] = [0.7, 0.7, 0.7]    # TN Gray
    else:
        for i in range(n):
            if preds[i] == 1:
                colors[i] = [0.0, 0.4, 1.0]    # Blue break
            else:
                colors[i] = [0.75, 0.75, 0.75]  # Gray original

    result = o3d.geometry.PointCloud()
    result.points = pcd.points
    result.colors = o3d.utility.Vector3dVector(colors)
    if pcd.has_normals():
        result.normals = pcd.normals

    o3d.io.write_point_cloud(output_path, result, write_ascii=False)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    return size_mb


def main():
    parser = argparse.ArgumentParser(description="Predict with PointNet++")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--voxel", type=float, default=config.VOXEL_SIZE)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=config.PRED_BATCH_SIZE)
    parser.add_argument("--gpu", type=str, default=config.DEVICE)
    args = parser.parse_args()

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")

    # Find checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")

    if not os.path.exists(ckpt_path):
        print(f"  ERROR: No checkpoint found at {ckpt_path}")
        print(f"  Train first: python3 train.py")
        return

    # Load model
    print(f"\n{'#'*70}")
    print("POINTNET++ PREDICTION")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Device: {device}")
    print(f"{'#'*70}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = checkpoint['config']

    model = PointNet2Classifier(
        input_channels=model_config['input_channels'],
        dropout=0.0).to(device)  # No dropout at inference
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Model loaded (epoch {checkpoint['epoch']}, val F1={checkpoint['val_f1']:.4f})")
    print(f"  Trained on: {model_config['training_fragments']}")

    # Find prediction files
    pred_dir = config.DATA_PRED_DIR
    files = sorted(glob.glob(os.path.join(pred_dir, "*.ply"))
                   + glob.glob(os.path.join(pred_dir, "*.PLY")))

    if not files:
        print(f"  No PLY files in {pred_dir}")
        return

    print(f"\n  Fragments to predict: {len(files)}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    all_stats = []

    for fi, fp in enumerate(files):
        frag_name = os.path.splitext(os.path.basename(fp))[0]
        print(f"\n{'='*70}")
        print(f"PREDICTING {fi+1}/{len(files)}: {frag_name}")
        print(f"{'='*70}")

        t0 = time.time()

        # Preprocess
        pcd, log = preprocess(fp, voxel_size=args.voxel)
        points = np.asarray(pcd.points).astype(np.float32)
        normals = np.asarray(pcd.normals).astype(np.float32)

        # Predict
        print(f"  Predicting {len(points):,} points...")
        proba = predict_fragment(
            points, normals, model, device,
            num_points=model_config['num_points'],
            batch_size=args.batch_size)

        # Post-process
        preds = postprocess(pcd, proba, threshold=args.threshold)

        # Save
        output_path = os.path.join(config.RESULTS_DIR, f"{frag_name}_predicted.ply")
        size_mb = save_result_ply(pcd, preds, output_path)
        print(f"  Saved: {output_path} ({size_mb:.1f} MB)")

        npz_path = os.path.join(config.RESULTS_DIR, f"{frag_name}_predictions.npz")
        np.savez(npz_path, predictions=preds, proba=proba)

        n_break = int(preds.sum())
        n_orig = len(preds) - n_break
        total_time = time.time() - t0

        print(f"  Result: {n_break:,} break ({100*n_break/len(preds):.1f}%), "
              f"{n_orig:,} original")
        print(f"  Time: {total_time:.1f}s")

        all_stats.append({
            'name': frag_name, 'n_points': len(preds),
            'n_break': n_break, 'n_original': n_orig,
            'time': total_time})

    # Summary
    print(f"\n{'#'*70}")
    print(f"PREDICTION COMPLETE - {len(all_stats)} fragments")
    print(f"{'#'*70}")
    print(f"\n  {'Fragment':<40} {'Points':>10} {'Break':>10} {'Break%':>8} {'Time':>8}")
    print(f"  {'-'*76}")
    for s in all_stats:
        pct = 100 * s['n_break'] / s['n_points']
        print(f"  {s['name']:<40} {s['n_points']:>10,} {s['n_break']:>10,} "
              f"{pct:>7.1f}% {s['time']:>7.1f}s")


if __name__ == "__main__":
    main()