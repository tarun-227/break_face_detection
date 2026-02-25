"""
Break Surface Detection - Improved Training Pipeline v2
=======================================================
Adds to the original pipeline:
  - Normal direction (nx, ny, nz)
  - New curvature features (anisotropy, omnivariance, gaussian_curvature, surface_variation)
  - Larger k values (k=50,75,100,150,200) for roughness, normal variance, PCA shape
  - Deeper RF (max_depth=15)

Usage:
  python train_v2.py GT_highres.ply --voxel 0.5
"""

import sys
import os
import time
import argparse
import numpy as np
import open3d as o3d
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, precision_score, recall_score, 
                             confusion_matrix)
import warnings
warnings.filterwarnings('ignore')


def extract_gt(colors):
    return (colors[:, 1] > 0.8) & (colors[:, 0] < 0.3) & (colors[:, 2] < 0.3)


def save_colored_ply(pcd, labels, preds, path):
    n = len(labels)
    c = np.zeros((n, 3))
    for i in range(n):
        if labels[i]==1 and preds[i]==1:   c[i]=[0,1,0]      # TP green
        elif labels[i]==0 and preds[i]==1: c[i]=[1,1,0]      # FP yellow
        elif labels[i]==1 and preds[i]==0: c[i]=[1,0,0]      # FN red
        else:                              c[i]=[0.7,0.7,0.7] # TN gray
    r = o3d.geometry.PointCloud()
    r.points = pcd.points
    r.colors = o3d.utility.Vector3dVector(c)
    if pcd.has_normals(): r.normals = pcd.normals
    o3d.io.write_point_cloud(path, r, write_ascii=False)
    return os.path.getsize(path)/(1024*1024)


def print_metrics(labels, preds, name):
    cm = confusion_matrix(labels, preds)
    tn,fp,fn,tp = cm.ravel()
    f1 = f1_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    print(f"    {name}:")
    print(f"      TP={tp:>10,}  FP={fp:>10,}  FN={fn:>10,}  TN={tn:>10,}")
    print(f"      F1={f1:.4f}  Precision={prec:.4f}  Recall={rec:.4f}")
    return {'f1':f1,'prec':prec,'rec':rec,'tp':tp,'fp':fp,'fn':fn,'tn':tn}


def iterative_fill(points, preds, tree, k=50, ratio=0.5, max_iter=10):
    p = preds.copy()
    for it in range(max_iter):
        orig_idx = np.where(p==0)[0]
        flips = []
        for idx in orig_idx:
            [cnt,nn,dsq] = tree.search_knn_vector_3d(points[idx], k+1)
            if p[np.array(nn[1:])].sum()/k >= ratio:
                flips.append(idx)
        for idx in flips:
            p[idx] = 1
        print(f"      Iter {it+1}: filled {len(flips):,}")
        if len(flips)==0: break
    return p


def main(filepath, voxel_size=0.5):
    
    basename = os.path.splitext(os.path.basename(filepath))[0]
    
    print(f"\n{'#'*75}")
    print("BREAK SURFACE DETECTION - IMPROVED PIPELINE v2")
    print(f"  New: normal direction, curvature features, larger k values, deeper RF")
    print(f"{'#'*75}")
    
    t_total = time.time()
    
    # ==================================================================
    # STEP 1: PREPROCESS
    # ==================================================================
    print(f"\n{'='*75}")
    print("STEP 1: PREPROCESS")
    print(f"{'='*75}")
    
    pcd = o3d.io.read_point_cloud(filepath)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    n_raw = len(pts)
    print(f"  Loaded: {n_raw:,}")
    
    _, uidx = np.unique(pts, axis=0, return_index=True)
    uidx = np.sort(uidx)
    pcd_c = o3d.geometry.PointCloud()
    pcd_c.points = o3d.utility.Vector3dVector(pts[uidx])
    pcd_c.colors = o3d.utility.Vector3dVector(cols[uidx])
    
    pcd_d = pcd_c.voxel_down_sample(voxel_size)
    cl, ind = pcd_d.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_f = pcd_d.select_by_index(ind)
    n_final = len(pcd_f.points)
    print(f"  After preprocessing: {n_final:,}")
    
    # ==================================================================
    # STEP 2: GT
    # ==================================================================
    print(f"\n{'='*75}")
    print("STEP 2: GROUND TRUTH")
    print(f"{'='*75}")
    
    labels = extract_gt(np.asarray(pcd_f.colors)).astype(int)
    n_break = labels.sum()
    print(f"  Break: {n_break:,} ({100*n_break/n_final:.2f}%)  Original: {n_final-n_break:,}")
    
    # ==================================================================
    # STEP 3: NORMALS
    # ==================================================================
    print(f"\n{'='*75}")
    print("STEP 3: NORMALS")
    print(f"{'='*75}")
    
    if n_final > 1_500_000:
        pcd_sm = pcd_f.voxel_down_sample(1.0)
        pcd_sm.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd_sm.orient_normals_consistent_tangent_plane(k=15)
        tree_sm = o3d.geometry.KDTreeFlann(pcd_sm)
        sm_n = np.asarray(pcd_sm.normals)
        fp = np.asarray(pcd_f.points)
        norms = np.zeros_like(fp)
        for i in range(n_final):
            [k2,idx,dsq] = tree_sm.search_knn_vector_3d(fp[i], 1)
            norms[i] = sm_n[idx[0]]
        pcd_f.normals = o3d.utility.Vector3dVector(norms)
    else:
        pcd_f.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd_f.orient_normals_consistent_tangent_plane(k=15)
    
    points = np.asarray(pcd_f.points)
    normals = np.asarray(pcd_f.normals)
    print(f"  Normals computed for {n_final:,} points")
    
    # ==================================================================
    # STEP 4: FEATURES
    # ==================================================================
    print(f"\n{'='*75}")
    print("STEP 4: COMPUTE FEATURES (expanded set)")
    print(f"{'='*75}")
    
    tree = o3d.geometry.KDTreeFlann(pcd_f)
    
    # Spacing
    dt = []
    for i in range(min(5000, n_final)):
        [k2,idx,dsq] = tree.search_knn_vector_3d(points[i], 2)
        dt.append(np.sqrt(dsq[1]))
    spacing = np.median(dt)
    print(f"  Spacing: {spacing:.4f}")
    
    # Define feature set
    # Original features
    k_density = [10, 30, 50, 100]
    r_multipliers = [2, 4, 6, 8]
    huang_scales = [1, 2, 3, 4, 5, 6]
    
    # Expanded k values for roughness, normal variance, PCA
    k_rough = [5, 10, 15, 20, 30, 50]
    k_nvar = [5, 10, 15, 20, 30, 50]
    k_pca = [10, 20, 30, 50]
    k_curv = [10, 20, 30, 50]
    
    feature_names = []
    
    # Density k-NN (original)
    for k in k_density:
        feature_names.extend([f"density_knn_k{k}", f"density_var_k{k}", f"density_std_k{k}"])
    # Density k-NN larger k
    for k in [150, 200]:
        feature_names.extend([f"density_knn_k{k}", f"density_var_k{k}", f"density_std_k{k}",
                              f"density_meandist_k{k}"])
    # Density radius
    for m in r_multipliers:
        feature_names.append(f"density_count_r{m}")
    # Larger radius
    for m in [10, 12, 15]:
        feature_names.append(f"density_count_r{m}")
    # Huang
    for s in huang_scales:
        feature_names.extend([f"huang_vol_s{s}", f"huang_voldist_s{s}"])
    # Larger huang scales
    for s in [7, 8, 9, 10]:
        feature_names.extend([f"huang_vol_s{s}", f"huang_voldist_s{s}"])
    feature_names.append("huang_sharpness")
    # Roughness (expanded)
    for k in k_rough:
        feature_names.append(f"roughness_k{k}")
    # Normal variance (expanded)
    for k in k_nvar:
        feature_names.append(f"normal_var_k{k}")
    # PCA shape (expanded) - linearity, planarity, scattering
    for k in k_pca:
        feature_names.extend([f"linearity_k{k}", f"planarity_k{k}", f"scattering_k{k}"])
    # NEW: Curvature features
    for k in k_curv:
        feature_names.extend([f"anisotropy_k{k}", f"omnivariance_k{k}", 
                              f"gaussian_curv_k{k}", f"surface_variation_k{k}",
                              f"min_curvature_k{k}", f"curvature_ratio_k{k}"])
    # NEW: Normal direction
    feature_names.extend(["nx", "ny", "nz"])
    # NEW: Normal deviation from local average
    for k in [10, 20, 30, 50]:
        feature_names.append(f"normal_deviation_k{k}")
    
    n_features = len(feature_names)
    print(f"  Total features: {n_features}")
    print(f"  Feature groups:")
    print(f"    Density k-NN:      {3*len(k_density) + 4*2} features")
    print(f"    Density radius:    {len(r_multipliers) + 3} features")
    print(f"    Huang:             {2*10 + 1} features")
    print(f"    Roughness:         {len(k_rough)} features")
    print(f"    Normal variance:   {len(k_nvar)} features")
    print(f"    PCA shape:         {3*len(k_pca)} features")
    print(f"    Curvature (NEW):   {6*len(k_curv)} features")
    print(f"    Normal dir (NEW):  3 features")
    print(f"    Normal dev (NEW):  4 features")
    
    features = np.zeros((n_final, n_features), dtype=np.float32)
    
    batch_size = 50000
    n_batches = (n_final + batch_size - 1) // batch_size
    
    t_feat = time.time()
    
    for bi in range(n_batches):
        start = bi * batch_size
        end = min(start + batch_size, n_final)
        
        if bi % 3 == 0:
            elapsed = time.time() - t_feat
            pct = 100 * start / n_final
            print(f"  Batch {bi+1}/{n_batches} ({pct:.0f}%, {elapsed:.0f}s elapsed)...")
        
        for pidx in range(start, end):
            fc = 0
            
            # ---- Density k-NN (original k values) ----
            for k in k_density:
                [cnt,nn,dsq] = tree.search_knn_vector_3d(points[pidx], k+1)
                dists = np.sqrt(np.array(dsq[1:]))
                md = dists[-1] if dists[-1]>0 else 1e-10
                features[pidx, fc] = k/(4/3*np.pi*md**3)
                features[pidx, fc+1] = np.var(dists)
                features[pidx, fc+2] = np.std(dists)
                fc += 3
            
            # ---- Density k-NN (larger k=150, 200) ----
            for k in [150, 200]:
                [cnt,nn,dsq] = tree.search_knn_vector_3d(points[pidx], k+1)
                dists = np.sqrt(np.array(dsq[1:]))
                md = dists[-1] if dists[-1]>0 else 1e-10
                features[pidx, fc] = k/(4/3*np.pi*md**3)
                features[pidx, fc+1] = np.var(dists)
                features[pidx, fc+2] = np.std(dists)
                features[pidx, fc+3] = np.mean(dists)
                fc += 4
            
            # ---- Density radius (original) ----
            for m in r_multipliers:
                r = spacing*m
                [cnt,nn2,dsq2] = tree.search_radius_vector_3d(points[pidx], r)
                features[pidx, fc] = cnt
                fc += 1
            
            # ---- Density radius (larger) ----
            for m in [10, 12, 15]:
                r = spacing*m
                [cnt,nn2,dsq2] = tree.search_radius_vector_3d(points[pidx], r)
                features[pidx, fc] = cnt
                fc += 1
            
            # ---- Huang (scales 1-6 original + 7-10 new) ----
            for s in list(range(1,7)) + [7,8,9,10]:
                r = spacing*s*2
                sv = 4/3*np.pi*r**3
                [cnt,nn2,dsq2] = tree.search_radius_vector_3d(points[pidx], r)
                features[pidx, fc] = cnt/sv if sv>0 else 0
                features[pidx, fc+1] = np.mean(np.sqrt(np.array(dsq2[1:]))) if cnt>1 else 0
                fc += 2
            
            # ---- Huang sharpness ----
            r_sh = spacing*4
            [cnt,nn2,dsq2] = tree.search_radius_vector_3d(points[pidx], r_sh)
            if cnt>3:
                np2 = points[np.array(nn2)]
                cov = np.cov(np2.T)
                eigs = np.sort(np.linalg.eigvalsh(cov))
                features[pidx, fc] = eigs[0]/(eigs[2]+1e-10)
            fc += 1
            
            # ---- Roughness (expanded k) ----
            for k in k_rough:
                [cnt,nn2,dsq2] = tree.search_knn_vector_3d(points[pidx], k+1)
                np2 = points[np.array(nn2[1:])]
                cov = np.cov(np2.T)
                eigs = np.sort(np.linalg.eigvalsh(cov))
                es = eigs.sum()
                features[pidx, fc] = eigs[0]/es if es>0 else 0
                fc += 1
            
            # ---- Normal variance (expanded k) ----
            for k in k_nvar:
                [cnt,nn2,dsq2] = tree.search_knn_vector_3d(points[pidx], k+1)
                features[pidx, fc] = np.var(normals[np.array(nn2[1:])])
                fc += 1
            
            # ---- PCA shape (expanded k) ----
            for k in k_pca:
                [cnt,nn2,dsq2] = tree.search_knn_vector_3d(points[pidx], k+1)
                np2 = points[np.array(nn2[1:])]
                cov = np.cov(np2.T)
                e = np.sort(np.linalg.eigvalsh(cov))[::-1]
                features[pidx, fc] = (e[0]-e[1])/(e[0]+1e-10)      # linearity
                features[pidx, fc+1] = (e[1]-e[2])/(e[0]+1e-10)    # planarity
                features[pidx, fc+2] = e[2]/(e[0]+1e-10)            # scattering
                fc += 3
            
            # ---- NEW: Curvature features (expanded k) ----
            for k in k_curv:
                [cnt,nn2,dsq2] = tree.search_knn_vector_3d(points[pidx], k+1)
                np2 = points[np.array(nn2[1:])]
                cov = np.cov(np2.T)
                e = np.sort(np.linalg.eigvalsh(cov))[::-1]
                e_sum = e.sum() + 1e-10
                
                features[pidx, fc] = (e[0]-e[2])/(e[0]+1e-10)       # anisotropy
                features[pidx, fc+1] = (e[0]*e[1]*e[2])**(1/3)      # omnivariance
                features[pidx, fc+2] = e[0]*e[2]                     # gaussian curvature
                features[pidx, fc+3] = e[2]/e_sum                    # surface variation
                features[pidx, fc+4] = e[2]                          # min curvature (eigenvalue)
                features[pidx, fc+5] = e[2]/(e[0]+1e-10)             # curvature ratio
                fc += 6
            
            # ---- NEW: Normal direction ----
            features[pidx, fc] = normals[pidx, 0]    # nx
            features[pidx, fc+1] = normals[pidx, 1]  # ny
            features[pidx, fc+2] = normals[pidx, 2]  # nz
            fc += 3
            
            # ---- NEW: Normal deviation from local average ----
            for k in [10, 20, 30, 50]:
                [cnt,nn2,dsq2] = tree.search_knn_vector_3d(points[pidx], k+1)
                avg_n = normals[np.array(nn2[1:])].mean(axis=0)
                avg_n_norm = np.linalg.norm(avg_n)
                if avg_n_norm > 1e-10:
                    avg_n /= avg_n_norm
                features[pidx, fc] = 1 - abs(np.dot(normals[pidx], avg_n))
                fc += 1
    
    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
    
    print(f"  Feature computation: {time.time()-t_feat:.1f}s")
    print(f"  Feature matrix: {features.shape}")
    
    # Verify feature count
    assert features.shape[1] == n_features, f"Mismatch: {features.shape[1]} vs {n_features}"
    
    # ==================================================================
    # STEP 5: TRAIN
    # ==================================================================
    print(f"\n{'='*75}")
    print("STEP 5: TRAIN (RF max_depth=15, 150 estimators)")
    print(f"{'='*75}")
    
    t_train = time.time()
    
    # 5-fold CV
    print(f"\n  5-Fold CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1, cv_prec, cv_rec = [], [], []
    
    for fold, (tr, te) in enumerate(skf.split(features, labels)):
        rf = RandomForestClassifier(n_estimators=150, max_depth=15,
                                     class_weight='balanced', random_state=42, n_jobs=-1)
        rf.fit(features[tr], labels[tr])
        p = rf.predict(features[te])
        f1 = f1_score(labels[te], p)
        prec = precision_score(labels[te], p)
        rec = recall_score(labels[te], p)
        cv_f1.append(f1); cv_prec.append(prec); cv_rec.append(rec)
        print(f"    Fold {fold+1}: F1={f1:.4f}  Prec={prec:.4f}  Recall={rec:.4f}")
    
    print(f"\n  CV: F1={np.mean(cv_f1):.4f}±{np.std(cv_f1):.4f}  "
          f"Prec={np.mean(cv_prec):.4f}  Recall={np.mean(cv_rec):.4f}")
    
    # Final model
    print(f"\n  Training final model...")
    rf_final = RandomForestClassifier(n_estimators=150, max_depth=15,
                                       class_weight='balanced', random_state=42, n_jobs=-1)
    rf_final.fit(features, labels)
    proba = rf_final.predict_proba(features)[:, 1]
    preds_raw = rf_final.predict(features)
    
    # Feature importance
    imp = rf_final.feature_importances_
    imp_order = np.argsort(imp)[::-1]
    
    print(f"\n  Feature Importance (Top 20):")
    print(f"    {'Rank':<5} {'Feature':<30} {'Importance':<12}")
    print(f"    {'-'*47}")
    for rank, idx in enumerate(imp_order[:20], 1):
        print(f"    {rank:<5} {feature_names[idx]:<30} {100*imp[idx]:.2f}%")
    
    # NEW vs OLD feature contribution
    new_features_set = set()
    for k in k_curv:
        new_features_set.update([f"anisotropy_k{k}", f"omnivariance_k{k}",
                                  f"gaussian_curv_k{k}", f"surface_variation_k{k}",
                                  f"min_curvature_k{k}", f"curvature_ratio_k{k}"])
    new_features_set.update(["nx", "ny", "nz"])
    for k in [10,20,30,50]:
        new_features_set.add(f"normal_deviation_k{k}")
    
    new_imp = sum(imp[i] for i,n in enumerate(feature_names) if n in new_features_set)
    old_imp = sum(imp[i] for i,n in enumerate(feature_names) if n not in new_features_set)
    print(f"\n  NEW features total importance: {100*new_imp:.1f}%")
    print(f"  OLD features total importance: {100*old_imp:.1f}%")
    
    print(f"\n  Training time: {time.time()-t_train:.1f}s")
    
    # ==================================================================
    # STEP 6: SAVE BEFORE POST-PROCESSING
    # ==================================================================
    before_path = f"{basename}_v2_before.ply"
    print(f"\n  Saving before post-processing → {before_path}")
    m_before = print_metrics(labels, preds_raw, "Raw RF output")
    save_colored_ply(pcd_f, labels, preds_raw, before_path)
    
    # ==================================================================
    # STEP 7: POST-PROCESSING
    # ==================================================================
    print(f"\n{'='*75}")
    print("STEP 7: POST-PROCESSING")
    print(f"{'='*75}")
    
    # Adaptive threshold
    print(f"\n  A. Adaptive Thresholding:")
    best_f1 = 0
    best_th = 0.5
    for th in np.arange(0.30, 0.80, 0.05):
        p = (proba >= th).astype(int)
        f1 = f1_score(labels, p)
        prec = precision_score(labels, p, zero_division=0)
        rec = recall_score(labels, p, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_th = th
        print(f"    th={th:.2f}: F1={f1:.4f}  Prec={prec:.4f}  Recall={rec:.4f}")
    
    print(f"    Best: {best_th:.2f}")
    preds = (proba >= best_th).astype(int)
    m_thresh = print_metrics(labels, preds, "After threshold")
    
    # Iterative filling
    print(f"\n  B. Iterative Filling:")
    print(f"    Pass 1: k=50, ratio=0.6")
    preds = iterative_fill(points, preds, tree, k=50, ratio=0.6, max_iter=10)
    m_fill1 = print_metrics(labels, preds, "After strict fill")
    
    print(f"    Pass 2: k=40, ratio=0.5")
    preds = iterative_fill(points, preds, tree, k=40, ratio=0.5, max_iter=10)
    m_fill2 = print_metrics(labels, preds, "After moderate fill")
    
    print(f"    Pass 3: k=30, ratio=0.4")
    preds = iterative_fill(points, preds, tree, k=30, ratio=0.4, max_iter=5)
    m_fill3 = print_metrics(labels, preds, "After relaxed fill")
    
    # Small cluster removal
    print(f"\n  C. Small Cluster Removal:")
    break_idx = np.where(preds==1)[0]
    if len(break_idx) > 0:
        bp = o3d.geometry.PointCloud()
        bp.points = o3d.utility.Vector3dVector(points[break_idx])
        cls = np.array(bp.cluster_dbscan(eps=spacing*3, min_points=10))
        from collections import Counter
        cc = Counter(cls[cls>=0])
        removed = 0
        small = set(c for c,cnt in cc.items() if cnt < 50)
        for i, cidx in enumerate(break_idx):
            if cls[i] in small or cls[i]==-1:
                preds[cidx] = 0; removed += 1
        print(f"    Removed {removed:,} noise points")
    
    m_final = print_metrics(labels, preds, "FINAL")
    
    # ==================================================================
    # STEP 8: SAVE AFTER POST-PROCESSING
    # ==================================================================
    after_path = f"{basename}_v2_after.ply"
    print(f"\n  Saving after post-processing → {after_path}")
    sz = save_colored_ply(pcd_f, labels, preds, after_path)
    print(f"  Saved ({sz:.1f} MB)")
    
    # Save predictions
    cm = confusion_matrix(labels, preds)
    tn,fp,fn,tp = cm.ravel()
    np.savez(f"{basename}_v2_predictions.npz",
             labels=labels, predictions=preds, proba=proba,
             tp=tp, fp=fp, fn=fn, tn=tn)
    
    # ==================================================================
    # SUMMARY
    # ==================================================================
    print(f"\n{'='*75}")
    print("RESULTS COMPARISON")
    print(f"{'='*75}")
    
    print(f"\n  {'Model':<25} {'F1':<10} {'Precision':<12} {'Recall':<10}")
    print(f"  {'-'*57}")
    print(f"  {'frag_1 (PDF)':<25} {'0.5686':<10} {'0.4809':<12} {'0.6955':<10}")
    print(f"  {'v1 (previous)':<25} {'0.6604':<10} {'0.8609':<12} {'0.5357':<10}")
    print(f"  {'v2 CV mean':<25} {np.mean(cv_f1):<10.4f} {np.mean(cv_prec):<12.4f} {np.mean(cv_rec):<10.4f}")
    print(f"  {'v2 final':<25} {m_final['f1']:<10.4f} {m_final['prec']:<12.4f} {m_final['rec']:<10.4f}")
    
    print(f"\n  Output files:")
    print(f"    {before_path} (before post-processing)")
    print(f"    {after_path} (after post-processing)")
    
    print(f"\n  Total time: {time.time()-t_total:.1f}s")
    print(f"{'='*75}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    parser.add_argument("--voxel", type=float, default=0.5)
    args = parser.parse_args()
    main(args.filepath, voxel_size=args.voxel)
