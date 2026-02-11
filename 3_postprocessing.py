# 3_postprocessing.py
"""
Break Surface Detection - Module 3: Post-processing
====================================================
Input:
  - {name}_probabilities.npy: Probability scores from algorithm
  - {name}_metadata.pkl: Metadata from preprocessing

Output:
  - {name}_final.ply: Final cleaned prediction with comparison colors

Processes (Option C - Minimal, avoiding buggy morphological/component steps):
1. Load probabilities and metadata
2. Adaptive thresholding (adjust threshold based on expected break surface size)
3. Interior filling (fill holes surrounded by predicted break surface)
4. Evaluate and save final result

Color coding in output:
- Green: True Positive (correctly detected break)
- Yellow: False Positive (incorrectly classified as break)
- Red: False Negative (missed break surface)
- Gray: True Negative (correctly classified as original)
"""

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import pickle
import sys
import time
from pathlib import Path


class PostProcessor:
    def __init__(self):
        """Initialize post-processor with fixed parameters"""
        # Adaptive thresholding parameters
        self.ADAPTIVE_PERCENTILE_BUFFER = 1.2  # Allow 20% more than GT size
        self.THRESHOLD_MIN = 0.3  # Minimum probability threshold
        self.THRESHOLD_MAX = 0.7  # Maximum probability threshold
        
        # Interior filling parameters
        self.FILL_INTERIOR_K = 30  # Number of neighbors to check
        self.FILL_INTERIOR_RATIO = 0.7  # Ratio of neighbors that must be break
        
        # Data storage
        self.pcd = None
        self.points = None
        self.normals = None
        self.gt_labels = None
        self.tree = None
        self.n_points = 0
        self.median_spacing = None
        
        # Probabilities
        self.probabilities = None
        
    def load_data(self, probabilities_npy, metadata_pkl):
        """
        Load probabilities and metadata
        
        Args:
            probabilities_npy: Path to probabilities numpy file
            metadata_pkl: Path to metadata pickle file
        """
        print(f"\n{'='*70}")
        print("STEP 1: LOADING DATA")
        print(f"{'='*70}")
        
        # Load probabilities
        print(f"Loading probabilities: {probabilities_npy}")
        self.probabilities = np.load(probabilities_npy)
        print(f"  Shape: {self.probabilities.shape}")
        print(f"  Range: [{self.probabilities.min():.3f}, {self.probabilities.max():.3f}]")
        print(f"  Mean: {self.probabilities.mean():.3f}")
        
        # Load metadata
        print(f"\nLoading metadata: {metadata_pkl}")
        with open(metadata_pkl, 'rb') as f:
            metadata = pickle.load(f)
        
        # Extract metadata
        self.n_points = metadata['n_points']
        self.median_spacing = metadata['median_spacing']
        self.gt_labels = metadata['gt_labels']
        self.points = metadata['points']
        self.normals = metadata['normals']
        
        print(f"\nMetadata loaded:")
        print(f"  Points: {self.n_points:,}")
        print(f"  Median spacing: {self.median_spacing:.6f}")
        print(f"  Break surface (GT): {self.gt_labels.sum():,} ({100*self.gt_labels.mean():.1f}%)")
        
        # Build KD-tree for spatial operations
        print(f"\nBuilding KD-tree...")
        t_start = time.time()
        self.tree = cKDTree(self.points)
        print(f"  Built in {time.time()-t_start:.1f}s")
        
        # Rebuild point cloud for visualization
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.points)
        self.pcd.normals = o3d.utility.Vector3dVector(self.normals)
    
    def analyze_break_surface(self):
        """
        Analyze ground truth break surface characteristics
        
        Used to set adaptive thresholds based on expected size.
        
        Returns:
            Dictionary of break surface statistics
        """
        print(f"\n{'='*70}")
        print("STEP 2: ANALYZING BREAK SURFACE CHARACTERISTICS")
        print(f"{'='*70}")
        
        break_points = self.points[self.gt_labels]
        n_break = len(break_points)
        
        # Estimate surface area using convex hull
        if n_break > 3:
            pcd_break = o3d.geometry.PointCloud()
            pcd_break.points = o3d.utility.Vector3dVector(break_points)
            hull, _ = pcd_break.compute_convex_hull()
            area = hull.get_surface_area()
        else:
            area = 0
        
        # Bounding box statistics
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(break_points)
        )
        bbox_volume = bbox.volume()
        bbox_dims = bbox.get_extent()
        
        # Compile statistics
        stats = {
            'n_points': n_break,
            'percentage': 100 * n_break / self.n_points,
            'area': area,
            'bbox_volume': bbox_volume,
            'bbox_dims': bbox_dims,
            'median_spacing': self.median_spacing,
        }
        
        print(f"\nBreak surface statistics:")
        print(f"  Points: {stats['n_points']:,} ({stats['percentage']:.1f}% of total)")
        print(f"  Estimated surface area: {stats['area']:.2f}")
        print(f"  Bounding box volume: {stats['bbox_volume']:.2f}")
        print(f"  Bounding box dimensions: [{bbox_dims[0]:.2f}, {bbox_dims[1]:.2f}, {bbox_dims[2]:.2f}]")
        
        return stats
    
    def adaptive_threshold(self, stats):
        """
        Apply adaptive probability threshold
        
        Goal: Find threshold that produces prediction size close to GT size.
        
        Strategy:
        1. Estimate target percentile based on expected break surface size
        2. Clamp threshold to reasonable range [0.3, 0.7]
        3. Apply threshold to probabilities
        
        Args:
            stats: Break surface statistics dictionary
            
        Returns:
            Binary mask of predicted break surface
        """
        print(f"\n{'='*70}")
        print("STEP 3: ADAPTIVE THRESHOLDING")
        print(f"{'='*70}")
        
        # Calculate target percentile
        # We want to keep approximately the same number of points as GT
        # but allow some buffer for uncertainty
        expected_percentage = stats['percentage'] * self.ADAPTIVE_PERCENTILE_BUFFER
        target_percentile = 100 - expected_percentage
        
        print(f"\nThreshold calculation:")
        print(f"  GT break surface: {stats['percentage']:.2f}% of points")
        print(f"  Target (with buffer): {expected_percentage:.2f}% of points")
        print(f"  Target percentile: {target_percentile:.2f}")
        
        # Find threshold at target percentile
        threshold = np.percentile(self.probabilities, target_percentile)
        
        # Clamp to reasonable range
        threshold = np.clip(threshold, self.THRESHOLD_MIN, self.THRESHOLD_MAX)
        
        print(f"  Computed threshold: {threshold:.3f}")
        print(f"  (clamped to [{self.THRESHOLD_MIN}, {self.THRESHOLD_MAX}])")
        
        # Apply threshold
        prediction = self.probabilities >= threshold
        n_predicted = prediction.sum()
        
        print(f"\nPrediction after adaptive threshold:")
        print(f"  Predicted break: {n_predicted:,} points ({100*n_predicted/self.n_points:.1f}%)")
        print(f"  Expected (GT): {stats['n_points']:,} points ({stats['percentage']:.1f}%)")
        print(f"  Ratio: {n_predicted/stats['n_points']:.2f}x GT size")
        
        return prediction, threshold
    
    def fill_interior(self, prediction):
        """
        Fill interior holes in predicted break surface
        
        Strategy:
        1. For each point NOT predicted as break
        2. Check k neighbors
        3. If majority of neighbors ARE predicted as break
        4. Fill this point (it's a hole surrounded by break surface)
        
        This addresses the issue where break surface edges are detected
        but interior is not filled.
        
        Args:
            prediction: Current binary prediction mask
            
        Returns:
            Filled binary prediction mask
        """
        print(f"\n{'='*70}")
        print("STEP 4: INTERIOR FILLING")
        print(f"{'='*70}")
        
        print(f"\nParameters:")
        print(f"  Neighbor count (k): {self.FILL_INTERIOR_K}")
        print(f"  Break ratio threshold: {self.FILL_INTERIOR_RATIO}")
        
        t_start = time.time()
        
        # Copy prediction to avoid modifying original
        filled = prediction.copy()
        
        # Find non-break points (potential holes)
        non_break_mask = ~prediction
        non_break_indices = np.where(non_break_mask)[0]
        
        print(f"\nAnalyzing {len(non_break_indices):,} non-break points...")
        
        # Query neighbors for all points
        _, indices = self.tree.query(
            self.points, 
            k=self.FILL_INTERIOR_K,
            workers=-1
        )
        
        # Check each non-break point
        fill_count = 0
        
        for idx in non_break_indices:
            # Get neighbors
            neighbor_indices = indices[idx]
            
            # Count how many neighbors are predicted as break
            neighbor_labels = prediction[neighbor_indices]
            break_ratio = neighbor_labels.sum() / self.FILL_INTERIOR_K
            
            # If surrounded by break surface, fill this point
            if break_ratio >= self.FILL_INTERIOR_RATIO:
                filled[idx] = True
                fill_count += 1
        
        print(f"\nInterior filling complete:")
        print(f"  Filled {fill_count:,} interior points")
        print(f"  Before: {prediction.sum():,} points")
        print(f"  After: {filled.sum():,} points")
        print(f"  Time: {time.time()-t_start:.1f}s")
        
        return filled
    
    def evaluate(self, prediction, label=""):
        """
        Evaluate prediction against ground truth
        
        Args:
            prediction: Binary prediction mask
            label: Label for this evaluation
            
        Returns:
            Dictionary of metrics
        """
        pred_mask = prediction.astype(bool)
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
        
        metrics = {
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'iou': iou,
            'n_predicted': pred_mask.sum(),
        }
        
        # Print evaluation
        if label:
            print(f"\n{'='*70}")
            print(f"EVALUATION: {label}")
            print(f"{'='*70}")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Positive (TP):   {tp:,}")
        print(f"  False Positive (FP):  {fp:,}")
        print(f"  False Negative (FN):  {fn:,}")
        print(f"  True Negative (TN):   {tn:,}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  IoU:       {iou:.4f}")
        
        return metrics
    
    def save_result(self, prediction, output_path):
        """
        Save final result with color-coded comparison
        
        Color coding:
        - Green: True Positive (correctly detected break)
        - Yellow: False Positive (incorrectly classified as break)
        - Red: False Negative (missed break surface)
        - Gray: True Negative (correctly classified as original)
        
        Args:
            prediction: Final binary prediction mask
            output_path: Path to save output PLY file
        """
        print(f"\n{'='*70}")
        print("STEP 5: SAVING FINAL RESULT")
        print(f"{'='*70}")
        
        # Create output point cloud
        pcd_out = o3d.geometry.PointCloud()
        pcd_out.points = self.pcd.points
        pcd_out.normals = self.pcd.normals
        
        # Color code based on TP/FP/FN/TN
        colors = np.zeros((self.n_points, 3))
        pred_mask = prediction.astype(bool)
        gt_mask = self.gt_labels.astype(bool)
        
        colors[(pred_mask & gt_mask)] = [0.0, 1.0, 0.0]    # TP = Green
        colors[(pred_mask & ~gt_mask)] = [1.0, 1.0, 0.0]   # FP = Yellow
        colors[(~pred_mask & gt_mask)] = [1.0, 0.0, 0.0]   # FN = Red
        colors[(~pred_mask & ~gt_mask)] = [0.5, 0.5, 0.5]  # TN = Gray
        
        pcd_out.colors = o3d.utility.Vector3dVector(colors)
        
        # Save
        print(f"\nSaving final result: {output_path}")
        t_start = time.time()
        o3d.io.write_point_cloud(str(output_path), pcd_out)
        print(f"  Saved ({time.time()-t_start:.1f}s)")
        
        # Print color legend
        print(f"\nColor legend:")
        print(f"  Green (TP):  {(pred_mask & gt_mask).sum():,} points - correctly detected break")
        print(f"  Yellow (FP): {(pred_mask & ~gt_mask).sum():,} points - incorrectly classified as break")
        print(f"  Red (FN):    {(~pred_mask & gt_mask).sum():,} points - missed break surface")
        print(f"  Gray (TN):   {(~pred_mask & ~gt_mask).sum():,} points - correctly classified as original")
    
    def run(self, probabilities_npy, metadata_pkl):
        """
        Run complete post-processing pipeline
        
        Args:
            probabilities_npy: Path to probabilities file
            metadata_pkl: Path to metadata file
        """
        print(f"\n{'='*70}")
        print("BREAK SURFACE DETECTION - POST-PROCESSING MODULE")
        print(f"{'='*70}")
        print(f"Input probabilities: {probabilities_npy}")
        print(f"Input metadata: {metadata_pkl}")
        
        # Step 1: Load data
        self.load_data(probabilities_npy, metadata_pkl)
        
        # Step 2: Analyze break surface
        stats = self.analyze_break_surface()
        
        # Step 3: Adaptive thresholding
        prediction_adaptive, threshold = self.adaptive_threshold(stats)
        metrics_adaptive = self.evaluate(prediction_adaptive, "Adaptive Threshold")
        
        # Step 4: Interior filling
        prediction_filled = self.fill_interior(prediction_adaptive)
        metrics_filled = self.evaluate(prediction_filled, "Interior Filled")
        
        # Final prediction is the filled version
        final_prediction = prediction_filled
        
        # Step 5: Save final result
        output_base = Path(probabilities_npy).stem.replace('_probabilities', '')
        output_path = f"{output_base}_final.ply"
        self.save_result(final_prediction, output_path)
        
        # Print summary
        print(f"\n{'='*70}")
        print("POST-PROCESSING COMPLETE")
        print(f"{'='*70}")
        
        print(f"\nResults Summary:")
        print(f"\n  After Adaptive Threshold:")
        print(f"    Threshold: {threshold:.3f}")
        print(f"    Precision: {metrics_adaptive['precision']:.4f}")
        print(f"    Recall:    {metrics_adaptive['recall']:.4f}")
        print(f"    F1 Score:  {metrics_adaptive['f1']:.4f}")
        
        print(f"\n  After Interior Filling:")
        print(f"    Precision: {metrics_filled['precision']:.4f}")
        print(f"    Recall:    {metrics_filled['recall']:.4f}")
        print(f"    F1 Score:  {metrics_filled['f1']:.4f}")
        
        print(f"\n  Improvement:")
        print(f"    F1 Score: {metrics_adaptive['f1']:.4f} → {metrics_filled['f1']:.4f} "
              f"({metrics_filled['f1'] - metrics_adaptive['f1']:+.4f})")
        
        print(f"\nOutput: {output_path}")


def main():
    """Main entry point"""
    
    if len(sys.argv) < 3:
        print("Usage: python 3_postprocessing.py <probabilities.npy> <metadata.pkl>")
        print("\nExample:")
        print("  python 3_postprocessing.py frag_1_probabilities.npy frag_1_metadata.pkl")
        print("\nOutput:")
        print("  {name}_final.ply - Final result with color-coded comparison")
        print("\nColor legend:")
        print("  Green (TP)  - Correctly detected break surface")
        print("  Yellow (FP) - Incorrectly classified as break")
        print("  Red (FN)    - Missed break surface")
        print("  Gray (TN)   - Correctly classified as original")
        sys.exit(1)
    
    probabilities_npy = sys.argv[1]
    metadata_pkl = sys.argv[2]
    
    if not Path(probabilities_npy).exists():
        print(f"ERROR: File not found: {probabilities_npy}")
        sys.exit(1)
    
    if not Path(metadata_pkl).exists():
        print(f"ERROR: File not found: {metadata_pkl}")
        sys.exit(1)
    
    # Run post-processing
    postprocessor = PostProcessor()
    postprocessor.run(probabilities_npy, metadata_pkl)


if __name__ == "__main__":
    main()