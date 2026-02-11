# 1_preprocessing.py
"""
Break Surface Detection - Module 1: Preprocessing
==================================================
Input: Raw PLY file (e.g., frag_1.ply)
Output: 
  - {name}_preprocessed.ply: Cleaned point cloud with normals
  - {name}_metadata.pkl: Spacing, GT labels, KD-tree index data

Processes:
1. Load raw point cloud
2. Truncate at corrupt vertex
3. Remove exact duplicate points
4. Statistical outlier removal
5. Compute median point spacing
6. Estimate normals
7. Extract ground truth labels from green paint annotation
8. Save preprocessed cloud and metadata
"""

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import pickle
import sys
import time
from pathlib import Path


class Preprocessor:
    def __init__(self):
        """Initialize preprocessor with fixed parameters"""
        # Configuration
        self.CORRUPT_VERTEX_IDX = 1887869
        self.OUTLIER_K = 20
        self.OUTLIER_STD = 2.0
        self.NORMAL_RADIUS_FACTOR = 8
        self.NORMAL_MAX_NN = 30
        self.GREEN_THRESHOLD = 0.4
        self.GREEN_RATIO = 1.2
        self.BRIGHT_GREEN_MIN = 0.5
        self.BRIGHT_GREEN_MAX = 0.5
        
        # Data storage
        self.pcd = None
        self.points = None
        self.normals = None
        self.colors = None
        self.gt_labels = None
        self.median_spacing = None
        self.n_points = 0
        
    def load_raw_pointcloud(self, ply_path):
        """
        Load raw point cloud from PLY file
        
        Args:
            ply_path: Path to input PLY file
            
        Returns:
            Point cloud object
        """
        print(f"\n{'='*70}")
        print("STEP 1: LOADING RAW POINT CLOUD")
        print(f"{'='*70}")
        
        print(f"Loading: {ply_path}")
        pcd = o3d.io.read_point_cloud(str(ply_path))
        n_raw = len(pcd.points)
        print(f"  Raw points loaded: {n_raw:,}")
        
        return pcd, n_raw
    
    def truncate_corrupt_data(self, pcd, n_raw):
        """
        Truncate point cloud at corrupt vertex index
        
        Known issue: PLY file has corrupt data at vertex 1,887,869
        
        Args:
            pcd: Point cloud object
            n_raw: Number of raw points
            
        Returns:
            Truncated point cloud
        """
        print(f"\n{'='*70}")
        print("STEP 2: TRUNCATING CORRUPT DATA")
        print(f"{'='*70}")
        
        if n_raw > self.CORRUPT_VERTEX_IDX:
            print(f"Truncating at corrupt vertex: {self.CORRUPT_VERTEX_IDX:,}")
            
            # Truncate points, colors, normals
            pcd.points = o3d.utility.Vector3dVector(
                np.asarray(pcd.points)[:self.CORRUPT_VERTEX_IDX]
            )
            pcd.colors = o3d.utility.Vector3dVector(
                np.asarray(pcd.colors)[:self.CORRUPT_VERTEX_IDX]
            )
            if pcd.has_normals():
                pcd.normals = o3d.utility.Vector3dVector(
                    np.asarray(pcd.normals)[:self.CORRUPT_VERTEX_IDX]
                )
            
            print(f"  After truncation: {len(pcd.points):,}")
        else:
            print("  No truncation needed")
        
        return pcd
    
    def remove_duplicates(self, pcd):
        """
        Remove exact duplicate points
        
        Duplicates occur when same 3D position appears multiple times.
        This also fixes the spacing calculation issue.
        
        Args:
            pcd: Point cloud object
            
        Returns:
            Point cloud with duplicates removed
        """
        print(f"\n{'='*70}")
        print("STEP 3: REMOVING DUPLICATE POINTS")
        print(f"{'='*70}")
        
        t_start = time.time()
        
        # Extract data
        points_raw = np.asarray(pcd.points)
        colors_raw = np.asarray(pcd.colors)
        
        # Find unique points (keeps first occurrence of each unique point)
        points_unique, unique_indices = np.unique(
            points_raw, 
            axis=0, 
            return_index=True
        )
        
        n_duplicates = len(points_raw) - len(points_unique)
        
        print(f"  Found {n_duplicates:,} exact duplicates ({time.time()-t_start:.1f}s)")
        print(f"  Unique points: {len(points_unique):,}")
        
        # Create new point cloud with unique points only
        pcd_unique = o3d.geometry.PointCloud()
        pcd_unique.points = o3d.utility.Vector3dVector(points_unique)
        pcd_unique.colors = o3d.utility.Vector3dVector(colors_raw[unique_indices])
        
        return pcd_unique
    
    def remove_outliers(self, pcd):
        """
        Remove statistical outliers
        
        Uses k-nearest neighbors to identify isolated points that are
        far from their neighbors (likely noise from scanning errors).
        
        Args:
            pcd: Point cloud object
            
        Returns:
            Point cloud with outliers removed
        """
        print(f"\n{'='*70}")
        print("STEP 4: REMOVING OUTLIERS")
        print(f"{'='*70}")
        
        print(f"  Parameters: k={self.OUTLIER_K}, std_ratio={self.OUTLIER_STD}")
        
        t_start = time.time()
        pcd_clean, inlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=self.OUTLIER_K,
            std_ratio=self.OUTLIER_STD
        )
        
        n_removed = len(pcd.points) - len(pcd_clean.points)
        print(f"  Removed {n_removed:,} outliers ({time.time()-t_start:.1f}s)")
        print(f"  Remaining points: {len(pcd_clean.points):,}")
        
        return pcd_clean
    
    def compute_spacing(self, points):
        """
        Compute median nearest-neighbor spacing
        
        This is critical for setting scales for normal estimation
        and multi-scale feature computation.
        
        Args:
            points: Nx3 array of point coordinates
            
        Returns:
            Median spacing value
        """
        print(f"\n{'='*70}")
        print("STEP 5: COMPUTING POINT SPACING")
        print(f"{'='*70}")
        
        t_start = time.time()
        
        # Build KD-tree for efficient nearest neighbor queries
        tree = cKDTree(points)
        
        # Sample points for speed (spacing is relatively uniform)
        sample_size = min(10000, len(points))
        sample_idx = np.random.choice(len(points), sample_size, replace=False)
        
        # Query k=2 to get nearest neighbor (k=1 is self)
        dists, _ = tree.query(
            points[sample_idx], 
            k=2, 
            workers=-1  # Use all CPU cores
        )
        
        # Extract nearest neighbor distances (second column)
        nn_dists = dists[:, 1]
        
        # Filter out any zeros (shouldn't happen after deduplication)
        nn_dists_filtered = nn_dists[nn_dists > 1e-10]
        
        if len(nn_dists_filtered) == 0:
            # Fallback: estimate from bounding box
            print("  WARNING: All NN distances are zero! Using fallback...")
            bbox_size = np.max(points.max(axis=0) - points.min(axis=0))
            median_spacing = bbox_size / (len(points) ** (1/3)) * 2
            print(f"  Fallback spacing: {median_spacing:.6f}")
        else:
            median_spacing = np.median(nn_dists_filtered)
            print(f"  Median spacing: {median_spacing:.6f}")
        
        # Print statistics for verification
        print(f"\n  Spacing statistics:")
        print(f"    Min:    {nn_dists_filtered.min():.6f}")
        print(f"    Median: {median_spacing:.6f}")
        print(f"    Mean:   {nn_dists_filtered.mean():.6f}")
        print(f"    Max:    {nn_dists_filtered.max():.6f}")
        print(f"  Computation time: {time.time()-t_start:.1f}s")
        
        return median_spacing, tree
    
    def estimate_normals(self, pcd, median_spacing):
        """
        Estimate surface normals at each point
        
        Uses hybrid search: ball-radius for scale + max neighbors cap.
        Normals are oriented toward camera at [0, 0, 1000].
        
        Args:
            pcd: Point cloud object
            median_spacing: Median point spacing
            
        Returns:
            Point cloud with normals estimated
        """
        print(f"\n{'='*70}")
        print("STEP 6: ESTIMATING NORMALS")
        print(f"{'='*70}")
        
        # Set search radius based on spacing
        radius = median_spacing * self.NORMAL_RADIUS_FACTOR
        
        print(f"  Normal radius: {radius:.6f} ({self.NORMAL_RADIUS_FACTOR}x spacing)")
        print(f"  Max neighbors: {self.NORMAL_MAX_NN}")
        
        t_start = time.time()
        
        # Estimate normals using hybrid search
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius,
                max_nn=self.NORMAL_MAX_NN
            )
        )
        
        # Orient normals consistently (toward camera)
        pcd.orient_normals_towards_camera_location(
            camera_location=np.array([0, 0, 1000])
        )
        
        print(f"  Normals estimated and oriented ({time.time()-t_start:.1f}s)")
        
        return pcd
    
    def extract_ground_truth(self, colors):
        """
        Extract ground truth labels from green paint annotation
        
        Break surface points are marked with bright green paint.
        Detection uses both color thresholds and brightness ratios.
        
        Args:
            colors: Nx3 array of RGB colors (0-1 range)
            
        Returns:
            Boolean array: True for break surface, False for original surface
        """
        print(f"\n{'='*70}")
        print("STEP 7: EXTRACTING GROUND TRUTH LABELS")
        print(f"{'='*70}")
        
        # Separate color channels
        r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
        
        # Criterion 1: Green is dominant and above threshold
        is_green = (
            (g > self.GREEN_THRESHOLD) & 
            (g > r * self.GREEN_RATIO) & 
            (g > b * self.GREEN_RATIO)
        )
        
        # Criterion 2: Bright green (for very bright annotations)
        is_bright = (
            (g > self.BRIGHT_GREEN_MIN) & 
            (r < self.BRIGHT_GREEN_MAX) & 
            (b < self.BRIGHT_GREEN_MAX)
        )
        
        # Combine criteria
        gt_labels = (is_green | is_bright)
        
        n_break = gt_labels.sum()
        n_original = (~gt_labels).sum()
        
        print(f"  Ground truth extracted:")
        print(f"    Break surface:    {n_break:,} points ({100*n_break/len(colors):.1f}%)")
        print(f"    Original surface: {n_original:,} points ({100*n_original/len(colors):.1f}%)")
        
        return gt_labels
    
    def save_outputs(self, pcd, metadata, output_base):
        """
        Save preprocessed point cloud and metadata
        
        Args:
            pcd: Preprocessed point cloud
            metadata: Dictionary containing all metadata
            output_base: Base name for output files (without extension)
        """
        print(f"\n{'='*70}")
        print("STEP 8: SAVING OUTPUTS")
        print(f"{'='*70}")
        
        # Save preprocessed point cloud
        ply_path = f"{output_base}_preprocessed.ply"
        print(f"Saving preprocessed point cloud: {ply_path}")
        t_start = time.time()
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"  Saved ({time.time()-t_start:.1f}s)")
        
        # Save metadata
        pkl_path = f"{output_base}_metadata.pkl"
        print(f"Saving metadata: {pkl_path}")
        t_start = time.time()
        with open(pkl_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"  Saved ({time.time()-t_start:.1f}s)")
        
        print(f"\nPreprocessing complete! Outputs:")
        print(f"  1. {ply_path}")
        print(f"  2. {pkl_path}")
    
    def run(self, ply_path):
        """
        Run complete preprocessing pipeline
        
        Args:
            ply_path: Path to input PLY file
        """
        print(f"\n{'='*70}")
        print("BREAK SURFACE DETECTION - PREPROCESSING MODULE")
        print(f"{'='*70}")
        print(f"Input: {ply_path}")
        
        # Step 1: Load raw point cloud
        pcd, n_raw = self.load_raw_pointcloud(ply_path)
        
        # Step 2: Truncate corrupt data
        pcd = self.truncate_corrupt_data(pcd, n_raw)
        
        # Step 3: Remove duplicates
        pcd = self.remove_duplicates(pcd)
        
        # Step 4: Remove outliers
        pcd = self.remove_outliers(pcd)
        
        # Extract data for further processing
        self.points = np.asarray(pcd.points)
        self.colors = np.asarray(pcd.colors)
        self.n_points = len(self.points)
        
        # Step 5: Compute spacing
        self.median_spacing, tree = self.compute_spacing(self.points)
        
        # Step 6: Estimate normals
        pcd = self.estimate_normals(pcd, self.median_spacing)
        self.normals = np.asarray(pcd.normals)
        self.pcd = pcd
        
        # Step 7: Extract ground truth
        self.gt_labels = self.extract_ground_truth(self.colors)
        
        # Prepare metadata for next module
        metadata = {
            'n_points': self.n_points,
            'median_spacing': self.median_spacing,
            'gt_labels': self.gt_labels,
            'points': self.points,
            'normals': self.normals,
            'colors': self.colors,
        }
        
        # Step 8: Save outputs
        output_base = Path(ply_path).stem
        self.save_outputs(pcd, metadata, output_base)
        
        print(f"\n{'='*70}")
        print("PREPROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"\nSummary:")
        print(f"  Input points:        {n_raw:,}")
        print(f"  Output points:       {self.n_points:,}")
        print(f"  Median spacing:      {self.median_spacing:.6f}")
        print(f"  Break surface (GT):  {self.gt_labels.sum():,} ({100*self.gt_labels.mean():.1f}%)")


def main():
    """Main entry point"""
    
    if len(sys.argv) < 2:
        print("Usage: python 1_preprocessing.py <input.ply>")
        print("\nExample:")
        print("  python 1_preprocessing.py frag_1.ply")
        print("\nOutputs:")
        print("  {name}_preprocessed.ply  - Cleaned point cloud with normals")
        print("  {name}_metadata.pkl      - Spacing, GT labels, and other metadata")
        sys.exit(1)
    
    ply_path = sys.argv[1]
    
    if not Path(ply_path).exists():
        print(f"ERROR: File not found: {ply_path}")
        sys.exit(1)
    
    # Run preprocessing
    preprocessor = Preprocessor()
    preprocessor.run(ply_path)


if __name__ == "__main__":
    main()