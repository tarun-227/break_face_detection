# 2_algorithm.py
"""
Break Surface Detection - Module 2: Algorithm
==============================================
Input: 
  - {name}_preprocessed.ply: Cleaned point cloud
  - {name}_metadata.pkl: Metadata from preprocessing

Output:
  - {name}_probabilities.npy: Probability scores for each point
  - {name}_features.npy: Feature matrix (optional, for debugging)
  - {name}_model.pkl: Trained Random Forest model
  - {name}_comparison.ply: Colored comparison with ground truth
  - {name}_metrics.txt: Performance metrics

Processes:
1. Load preprocessed data
2. Compute density features (k-NN and radius-based)
3. Compute Huang integral invariant features
4. Compute geometric features (normal variance)
5. Compute roughness features
6. Train Random Forest classifier
7. Generate predictions and probabilities
8. Evaluate against ground truth
9. Save comparison visualization and metrics
"""

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import sys
import time
from pathlib import Path


class BreakSurfaceAlgorithm:
    def __init__(self):
        """Initialize algorithm with fixed parameters"""
        # Feature computation parameters
        self.BATCH_SIZE = 100000
        self.NUM_WORKERS = -1  # Use all CPU cores
        
        # Density features
        self.DENSITY_K_VALUES = [10, 20, 30, 50, 100]
        self.DENSITY_R_FACTORS = [2, 4, 6, 8]
        
        # Huang features
        self.HUANG_SCALES = 6
        self.HUANG_MIN_SCALE = 2
        self.HUANG_MAX_SCALE = 8
        
        # Geometric features
        self.NORMAL_VAR_K_VALUES = [10, 20, 30]
        
        # Roughness features
        self.ROUGHNESS_K_VALUES = [10, 20, 30]
        
        # Random Forest parameters
        self.RF_N_ESTIMATORS = 100
        self.RF_MAX_DEPTH = 10
        self.RF_MIN_SAMPLES_SPLIT = 100
        self.RF_MIN_SAMPLES_LEAF = 50
        self.RF_CLASS_WEIGHT = 'balanced'
        self.RF_RANDOM_STATE = 42
        
        # Data storage
        self.pcd = None
        self.points = None
        self.normals = None
        self.colors = None
        self.gt_labels = None
        self.tree = None
        self.median_spacing = None
        self.n_points = 0
        
        # Feature storage
        self.feature_names = []
        self.feature_matrix = None
        
        # Model storage
        self.rf = None
        self.scaler = None
        
    def load_preprocessed_data(self, preprocessed_ply, metadata_pkl):
        """
        Load preprocessed point cloud and metadata
        
        Args:
            preprocessed_ply: Path to preprocessed PLY file
            metadata_pkl: Path to metadata pickle file
        """
        print(f"\n{'='*70}")
        print("STEP 1: LOADING PREPROCESSED DATA")
        print(f"{'='*70}")
        
        # Load point cloud
        print(f"Loading point cloud: {preprocessed_ply}")
        self.pcd = o3d.io.read_point_cloud(str(preprocessed_ply))
        print(f"  Points: {len(self.pcd.points):,}")
        
        # Load metadata
        print(f"Loading metadata: {metadata_pkl}")
        with open(metadata_pkl, 'rb') as f:
            metadata = pickle.load(f)
        
        # Extract metadata
        self.n_points = metadata['n_points']
        self.median_spacing = metadata['median_spacing']
        self.gt_labels = metadata['gt_labels']
        self.points = metadata['points']
        self.normals = metadata['normals']
        self.colors = metadata['colors']
        
        print(f"\nLoaded metadata:")
        print(f"  Points: {self.n_points:,}")
        print(f"  Median spacing: {self.median_spacing:.6f}")
        print(f"  Break surface (GT): {self.gt_labels.sum():,} ({100*self.gt_labels.mean():.1f}%)")
        
        # Build KD-tree for feature computation
        print(f"\nBuilding KD-tree...")
        t_start = time.time()
        self.tree = cKDTree(self.points)
        print(f"  Built in {time.time()-t_start:.1f}s")
    
    def _add_feature(self, name, values):
        """
        Add a feature to the feature matrix
        
        Args:
            name: Feature name
            values: Nx1 array of feature values
        """
        if self.feature_matrix is None:
            self.feature_matrix = values.reshape(-1, 1)
        else:
            self.feature_matrix = np.column_stack([self.feature_matrix, values])
        
        self.feature_names.append(name)
    
    def compute_density_features(self):
        """
        Compute density-based features
        
        Two types:
        1. k-NN density: mean/std/max/var of distances to k nearest neighbors
        2. Radius density: count of points within fixed radius
        
        These capture local point density which differs between
        break (smoother/denser) and original (weathered/sparser) surfaces.
        """
        print(f"\n{'='*70}")
        print("STEP 2: COMPUTING DENSITY FEATURES")
        print(f"{'='*70}")
        
        batch_size = self.BATCH_SIZE
        n_batches = int(np.ceil(self.n_points / batch_size))
        
        # k-NN based density features
        print(f"\nk-NN density features:")
        for k in self.DENSITY_K_VALUES:
            t_start = time.time()
            
            # Pre-allocate arrays
            density_mean = np.zeros(self.n_points)
            density_std = np.zeros(self.n_points)
            density_max = np.zeros(self.n_points)
            density_var = np.zeros(self.n_points)
            
            # Process in batches
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, self.n_points)
                
                # Query k nearest neighbors
                batch_points = self.points[start_idx:end_idx]
                dists, _ = self.tree.query(
                    batch_points, 
                    k=k, 
                    workers=self.NUM_WORKERS
                )
                
                # Compute statistics
                density_mean[start_idx:end_idx] = dists.mean(axis=1)
                density_std[start_idx:end_idx] = dists.std(axis=1)
                density_max[start_idx:end_idx] = dists.max(axis=1)
                density_var[start_idx:end_idx] = dists.var(axis=1)
            
            # Add features
            self._add_feature(f'density_mean_k{k}', density_mean)
            self._add_feature(f'density_std_k{k}', density_std)
            self._add_feature(f'density_max_k{k}', density_max)
            self._add_feature(f'density_var_k{k}', density_var)
            
            print(f"  k={k}: {time.time()-t_start:.1f}s")
        
        # Radius-based density features
        print(f"\nRadius-based density features:")
        for r_factor in self.DENSITY_R_FACTORS:
            radius = self.median_spacing * r_factor
            t_start = time.time()
            
            density_count = np.zeros(self.n_points)
            
            # Process in batches
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, self.n_points)
                
                # Query points within radius
                batch_points = self.points[start_idx:end_idx]
                indices_list = self.tree.query_ball_point(
                    batch_points, 
                    radius, 
                    workers=self.NUM_WORKERS
                )
                counts = np.array([len(idx) for idx in indices_list])
                density_count[start_idx:end_idx] = counts
            
            self._add_feature(f'density_count_r{r_factor}', density_count)
            
            print(f"  r={r_factor}x spacing: {time.time()-t_start:.1f}s")
    
    def compute_huang_features(self):
        """
        Compute Huang et al. integral invariant features
        
        From SIGGRAPH 2006: "Reassembling Fractured Objects by Geometric Matching"
        
        Features:
        1. Volume descriptor: ball-domain intersection ratio (related to mean curvature)
        2. Volume distance descriptor: weighted distance integral (related to curvature difference)
        3. Surface sharpness: RMS deviation from planarity across scales
        
        These are multi-scale geometric descriptors that capture
        surface characteristics independent of orientation.
        """
        print(f"\n{'='*70}")
        print("STEP 3: COMPUTING HUANG INTEGRAL INVARIANT FEATURES")
        print(f"{'='*70}")
        
        batch_size = self.BATCH_SIZE
        n_batches = int(np.ceil(self.n_points / batch_size))
        
        # Compute multi-scale radii
        n_scales = self.HUANG_SCALES
        scales = np.linspace(
            self.median_spacing * self.HUANG_MIN_SCALE,
            self.median_spacing * self.HUANG_MAX_SCALE,
            n_scales
        )
        
        print(f"\nUsing {n_scales} scales:")
        for i, r in enumerate(scales):
            print(f"  r[{i}] = {r:.4f} ({r/self.median_spacing:.1f}x spacing)")
        
        # Volume descriptors
        print(f"\nVolume descriptors:")
        volume_features = []
        
        for scale_idx, radius in enumerate(scales):
            t_start = time.time()
            
            volume_desc = np.zeros(self.n_points)
            
            # Process in batches
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, self.n_points)
                
                batch_points = self.points[start_idx:end_idx]
                
                # Count neighbors within radius
                indices_list = self.tree.query_ball_point(
                    batch_points, 
                    radius, 
                    workers=self.NUM_WORKERS
                )
                counts = np.array([len(idx) for idx in indices_list], dtype=float)
                volume_desc[start_idx:end_idx] = counts
            
            # Normalize by 95th percentile (robust to outliers)
            max_count = np.percentile(volume_desc, 95)
            volume_desc = volume_desc / (max_count + 1e-10)
            volume_desc = np.clip(volume_desc, 0, 1)
            
            self._add_feature(f'huang_volume_r{scale_idx}', volume_desc)
            volume_features.append(volume_desc)
            
            print(f"  scale {scale_idx}: {time.time()-t_start:.1f}s")
        
        # Surface sharpness (multi-scale RMS deviation from planarity)
        print(f"\nSurface sharpness:")
        vol_array = np.array(volume_features)
        deviations = (vol_array - 0.5) ** 2  # Deviation from planar value (0.5)
        sharpness = np.sqrt(deviations.mean(axis=0))
        self._add_feature('huang_sharpness', sharpness)
        print(f"  Computed")
        
        # Volume distance descriptors
        print(f"\nVolume distance descriptors:")
        for scale_idx, radius in enumerate(scales):
            t_start = time.time()
            
            vd_desc = np.zeros(self.n_points)
            
            # Process in batches
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, self.n_points)
                
                batch_points = self.points[start_idx:end_idx]
                
                # Query neighbors within radius (up to 100 points)
                dists, _ = self.tree.query(
                    batch_points,
                    k=100,
                    distance_upper_bound=radius,
                    workers=self.NUM_WORKERS
                )
                
                # Compute mean squared distance for each point
                for i in range(len(batch_points)):
                    valid = dists[i] < np.inf
                    if valid.sum() > 0:
                        sq_dists = dists[i][valid] ** 2
                        vd_desc[start_idx + i] = sq_dists.mean()
            
            # Normalize by radius squared
            vd_desc = vd_desc / (radius ** 2 + 1e-10)
            
            self._add_feature(f'huang_voldist_r{scale_idx}', vd_desc)
            
            print(f"  scale {scale_idx}: {time.time()-t_start:.1f}s")
    
    def compute_geometric_features(self):
        """
        Compute geometric features based on normal variation
        
        Normal variance captures surface curvature and irregularity.
        Break surfaces tend to be smoother (lower variance) than
        weathered original surfaces.
        """
        print(f"\n{'='*70}")
        print("STEP 4: COMPUTING GEOMETRIC FEATURES")
        print(f"{'='*70}")
        
        batch_size = self.BATCH_SIZE
        n_batches = int(np.ceil(self.n_points / batch_size))
        
        print(f"\nNormal variance features:")
        for k in self.NORMAL_VAR_K_VALUES:
            t_start = time.time()
            
            normal_var = np.zeros(self.n_points)
            
            # Process in batches
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, self.n_points)
                
                batch_points = self.points[start_idx:end_idx]
                
                # Query k nearest neighbors
                _, indices = self.tree.query(
                    batch_points, 
                    k=k, 
                    workers=self.NUM_WORKERS
                )
                
                # Compute normal variance for each point
                for i in range(len(batch_points)):
                    nb_normals = self.normals[indices[i]]
                    # Sum of variances across x, y, z components
                    normal_var[start_idx + i] = nb_normals.var(axis=0).sum()
            
            self._add_feature(f'normal_variance_k{k}', normal_var)
            
            print(f"  k={k}: {time.time()-t_start:.1f}s")
    
    def compute_roughness_features(self):
        """
        Compute surface roughness features (Huang et al. Eq. 5-6)
        
        Local bending energy: measures how much normals change
        relative to distance. Higher values indicate rougher surfaces.
        
        Roughness = mean((n_i - n_j)^2 / ||p_i - p_j||^2)
        """
        print(f"\n{'='*70}")
        print("STEP 5: COMPUTING ROUGHNESS FEATURES")
        print(f"{'='*70}")
        
        batch_size = self.BATCH_SIZE
        n_batches = int(np.ceil(self.n_points / batch_size))
        
        print(f"\nRoughness features:")
        for k in self.ROUGHNESS_K_VALUES:
            t_start = time.time()
            
            roughness = np.zeros(self.n_points)
            
            # Process in batches
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, self.n_points)
                
                batch_points = self.points[start_idx:end_idx]
                
                # Query k+1 neighbors (k=1 is self, exclude it)
                dists, indices = self.tree.query(
                    batch_points, 
                    k=k+1, 
                    workers=self.NUM_WORKERS
                )
                
                # Exclude self (first column)
                dists = dists[:, 1:]
                indices = indices[:, 1:]
                
                # Compute roughness for each point
                for i in range(len(batch_points)):
                    nb_normals = self.normals[indices[i]]
                    nb_dists = dists[i]
                    
                    # Normal difference squared
                    normal_diffs = np.linalg.norm(
                        self.normals[start_idx + i] - nb_normals, 
                        axis=1
                    ) ** 2
                    
                    # Distance squared (add epsilon to avoid division by zero)
                    dist_sq = nb_dists ** 2 + 1e-8
                    
                    # Local bending energy
                    roughness[start_idx + i] = (normal_diffs / dist_sq).mean()
            
            self._add_feature(f'roughness_k{k}', roughness)
            
            print(f"  k={k}: {time.time()-t_start:.1f}s")
    
    def train_random_forest(self):
        """
        Train Random Forest classifier
        
        Random Forest is chosen for:
        1. Handles non-linear relationships
        2. Feature importance ranking
        3. No assumption of feature distributions
        4. Robust to outliers
        5. Class imbalance handling (balanced weights)
        """
        print(f"\n{'='*70}")
        print("STEP 6: TRAINING RANDOM FOREST CLASSIFIER")
        print(f"{'='*70}")
        
        print(f"\nFeature matrix shape: {self.feature_matrix.shape}")
        print(f"Total features: {len(self.feature_names)}")
        
        # Standardize features (zero mean, unit variance)
        print(f"\nStandardizing features...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.feature_matrix)
        
        # Prepare labels
        y = self.gt_labels.astype(int)
        
        # Train Random Forest
        print(f"\nTraining Random Forest:")
        print(f"  n_estimators: {self.RF_N_ESTIMATORS}")
        print(f"  max_depth: {self.RF_MAX_DEPTH}")
        print(f"  min_samples_split: {self.RF_MIN_SAMPLES_SPLIT}")
        print(f"  min_samples_leaf: {self.RF_MIN_SAMPLES_LEAF}")
        print(f"  class_weight: {self.RF_CLASS_WEIGHT}")
        
        t_start = time.time()
        
        self.rf = RandomForestClassifier(
            n_estimators=self.RF_N_ESTIMATORS,
            max_depth=self.RF_MAX_DEPTH,
            min_samples_split=self.RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=self.RF_MIN_SAMPLES_LEAF,
            class_weight=self.RF_CLASS_WEIGHT,
            random_state=self.RF_RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        
        self.rf.fit(X_scaled, y)
        
        print(f"\n  Training completed in {time.time()-t_start:.1f}s")
        
        # Print feature importance
        importances = self.rf.feature_importances_
        feature_importance = sorted(
            zip(self.feature_names, importances), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        print(f"\nTop 10 most important features:")
        for rank, (name, importance) in enumerate(feature_importance[:10], 1):
            print(f"  {rank:2d}. {name:<30} {importance:.4f}")
    
    def predict_and_evaluate(self):
        """
        Generate predictions and evaluate against ground truth
        
        Returns:
            predictions: Binary predictions (0/1)
            probabilities: Probability scores (0-1)
            metrics: Dictionary of evaluation metrics
        """
        print(f"\n{'='*70}")
        print("STEP 7: PREDICTION AND EVALUATION")
        print(f"{'='*70}")
        
        # Standardize features
        X_scaled = self.scaler.transform(self.feature_matrix)
        
        # Predict
        print(f"\nGenerating predictions...")
        t_start = time.time()
        predictions = self.rf.predict(X_scaled)
        probabilities = self.rf.predict_proba(X_scaled)[:, 1]  # Probability of class 1 (break)
        print(f"  Time: {time.time()-t_start:.1f}s")
        
        print(f"\nProbability statistics:")
        print(f"  Min:    {probabilities.min():.3f}")
        print(f"  Max:    {probabilities.max():.3f}")
        print(f"  Mean:   {probabilities.mean():.3f}")
        print(f"  Median: {np.median(probabilities):.3f}")
        
        # Evaluate against ground truth
        print(f"\n{'='*70}")
        print("EVALUATION AGAINST GROUND TRUTH")
        print(f"{'='*70}")
        
        pred_mask = predictions.astype(bool)
        gt_mask = self.gt_labels.astype(bool)
        
        # Confusion matrix
        tp = (pred_mask & gt_mask).sum()
        fp = (pred_mask & ~gt_mask).sum()
        fn = (~pred_mask & gt_mask).sum()
        tn = (~pred_mask & ~gt_mask).sum()
        
        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / self.n_points
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        print(f"\nConfusion Matrix:")
        print(f"  True Positive (TP):   {tp:,} (correctly detected break)")
        print(f"  False Positive (FP):  {fp:,} (incorrectly classified as break)")
        print(f"  False Negative (FN):  {fn:,} (missed break surface)")
        print(f"  True Negative (TN):   {tn:,} (correctly classified as original)")
        
        print(f"\nPerformance Metrics:")
        print(f"  Precision: {precision:.4f} ({100*precision:.1f}% of predictions are correct)")
        print(f"  Recall:    {recall:.4f} ({100*recall:.1f}% of break surface detected)")
        print(f"  F1 Score:  {f1:.4f} (harmonic mean of precision and recall)")
        print(f"  Accuracy:  {accuracy:.4f} ({100*accuracy:.1f}% overall correct)")
        print(f"  IoU:       {iou:.4f} (intersection over union)")
        
        metrics = {
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'iou': iou,
            'n_predicted': pred_mask.sum(),
            'n_gt': gt_mask.sum(),
        }
        
        return predictions, probabilities, metrics
    
    def save_comparison(self, predictions, output_base):
        """
        Save colored comparison with ground truth
        
        Color coding:
        - Green: True Positive (correctly detected break)
        - Yellow: False Positive (incorrectly classified as break)
        - Red: False Negative (missed break surface)
        - Gray: True Negative (correctly classified as original)
        
        Args:
            predictions: Binary predictions
            output_base: Base name for output file
        """
        print(f"\n{'='*70}")
        print("STEP 8: SAVING COMPARISON VISUALIZATION")
        print(f"{'='*70}")
        
        # Create colored point cloud
        pcd_comp = o3d.geometry.PointCloud()
        pcd_comp.points = self.pcd.points
        pcd_comp.normals = self.pcd.normals
        
        # Assign colors based on prediction vs ground truth
        colors_comp = np.zeros((self.n_points, 3))
        pred_mask = predictions.astype(bool)
        gt_mask = self.gt_labels.astype(bool)
        
        colors_comp[(pred_mask & gt_mask)] = [0.0, 1.0, 0.0]    # TP = Green
        colors_comp[(pred_mask & ~gt_mask)] = [1.0, 1.0, 0.0]   # FP = Yellow
        colors_comp[(~pred_mask & gt_mask)] = [1.0, 0.0, 0.0]   # FN = Red
        colors_comp[(~pred_mask & ~gt_mask)] = [0.5, 0.5, 0.5]  # TN = Gray
        
        pcd_comp.colors = o3d.utility.Vector3dVector(colors_comp)
        
        # Save
        output_path = f"{output_base}_comparison.ply"
        print(f"\nSaving comparison: {output_path}")
        t_start = time.time()
        o3d.io.write_point_cloud(output_path, pcd_comp)
        print(f"  Saved ({time.time()-t_start:.1f}s)")
        
        print(f"\nColor legend:")
        print(f"  Green (TP):  {(pred_mask & gt_mask).sum():,} points")
        print(f"  Yellow (FP): {(pred_mask & ~gt_mask).sum():,} points")
        print(f"  Red (FN):    {(~pred_mask & gt_mask).sum():,} points")
        print(f"  Gray (TN):   {(~pred_mask & ~gt_mask).sum():,} points")
    
    def save_outputs(self, probabilities, predictions, metrics, output_base):
        """
        Save all outputs
        
        Args:
            probabilities: Probability scores
            predictions: Binary predictions
            metrics: Performance metrics
            output_base: Base name for output files
        """
        print(f"\n{'='*70}")
        print("STEP 9: SAVING OUTPUTS")
        print(f"{'='*70}")
        
        # Save probabilities
        prob_path = f"{output_base}_probabilities.npy"
        print(f"\nSaving probabilities: {prob_path}")
        np.save(prob_path, probabilities)
        print(f"  Saved")
        
        # Save feature matrix (optional, for debugging)
        features_path = f"{output_base}_features.npy"
        print(f"\nSaving features: {features_path}")
        np.save(features_path, self.feature_matrix)
        print(f"  Saved (shape: {self.feature_matrix.shape})")
        
        # Save model
        model_path = f"{output_base}_model.pkl"
        print(f"\nSaving model: {model_path}")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'rf': self.rf,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
        print(f"  Saved")
        
        # Save metrics
        metrics_path = f"{output_base}_metrics.txt"
        print(f"\nSaving metrics: {metrics_path}")
        with open(metrics_path, 'w') as f:
            f.write("Break Surface Detection - Algorithm Results\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write(f"  True Positive (TP):   {metrics['tp']:,}\n")
            f.write(f"  False Positive (FP):  {metrics['fp']:,}\n")
            f.write(f"  False Negative (FN):  {metrics['fn']:,}\n")
            f.write(f"  True Negative (TN):   {metrics['tn']:,}\n\n")
            
            f.write("Performance Metrics:\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1 Score:  {metrics['f1']:.4f}\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  IoU:       {metrics['iou']:.4f}\n\n")
            
            f.write("Prediction Summary:\n")
            f.write(f"  Predicted break:  {metrics['n_predicted']:,}\n")
            f.write(f"  Ground truth:     {metrics['n_gt']:,}\n")
        print(f"  Saved")
        
        print(f"\nAll outputs saved:")
        print(f"  1. {prob_path}")
        print(f"  2. {features_path}")
        print(f"  3. {model_path}")
        print(f"  4. {metrics_path}")
        print(f"  5. {output_base}_comparison.ply")
    
    def run(self, preprocessed_ply, metadata_pkl):
        """
        Run complete algorithm pipeline
        
        Args:
            preprocessed_ply: Path to preprocessed PLY file
            metadata_pkl: Path to metadata pickle file
        """
        print(f"\n{'='*70}")
        print("BREAK SURFACE DETECTION - ALGORITHM MODULE")
        print(f"{'='*70}")
        print(f"Input PLY: {preprocessed_ply}")
        print(f"Input metadata: {metadata_pkl}")
        
        # Step 1: Load data
        self.load_preprocessed_data(preprocessed_ply, metadata_pkl)
        
        # Step 2-5: Compute features
        self.compute_density_features()
        self.compute_huang_features()
        self.compute_geometric_features()
        self.compute_roughness_features()
        
        print(f"\n{'='*70}")
        print(f"FEATURE COMPUTATION COMPLETE")
        print(f"{'='*70}")
        print(f"  Feature matrix shape: {self.feature_matrix.shape}")
        print(f"  Total features: {len(self.feature_names)}")
        
        # Step 6: Train model
        self.train_random_forest()
        
        # Step 7: Predict and evaluate
        predictions, probabilities, metrics = self.predict_and_evaluate()
        
        # Step 8: Save comparison
        output_base = Path(preprocessed_ply).stem.replace('_preprocessed', '')
        self.save_comparison(predictions, output_base)
        
        # Step 9: Save all outputs
        self.save_outputs(probabilities, predictions, metrics, output_base)
        
        print(f"\n{'='*70}")
        print("ALGORITHM MODULE COMPLETE")
        print(f"{'='*70}")
        print(f"\nFinal Results:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")


def main():
    """Main entry point"""
    
    if len(sys.argv) < 3:
        print("Usage: python 2_algorithm.py <preprocessed.ply> <metadata.pkl>")
        print("\nExample:")
        print("  python 2_algorithm.py frag_1_preprocessed.ply frag_1_metadata.pkl")
        print("\nOutputs:")
        print("  {name}_probabilities.npy  - Probability scores")
        print("  {name}_features.npy       - Feature matrix")
        print("  {name}_model.pkl          - Trained model")
        print("  {name}_comparison.ply     - Colored comparison with GT")
        print("  {name}_metrics.txt        - Performance metrics")
        sys.exit(1)
    
    preprocessed_ply = sys.argv[1]
    metadata_pkl = sys.argv[2]
    
    if not Path(preprocessed_ply).exists():
        print(f"ERROR: File not found: {preprocessed_ply}")
        sys.exit(1)
    
    if not Path(metadata_pkl).exists():
        print(f"ERROR: File not found: {metadata_pkl}")
        sys.exit(1)
    
    # Run algorithm
    algorithm = BreakSurfaceAlgorithm()
    algorithm.run(preprocessed_ply, metadata_pkl)


if __name__ == "__main__":
    main()