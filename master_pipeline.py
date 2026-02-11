# master_pipeline.py
"""
Break Surface Detection - Master Pipeline
==========================================
Runs all three modules in sequence:
1. Preprocessing
2. Algorithm (Feature Computation + Classification)
3. Post-processing

Usage:
    python3 master_pipeline.py <input.ply>

Example:
    python3 master_pipeline.py frag_1.ply

Outputs:
    Preprocessing:
        - {name}_preprocessed.ply
        - {name}_metadata.pkl
    
    Algorithm:
        - {name}_probabilities.npy
        - {name}_features.npy
        - {name}_model.pkl
        - {name}_comparison.ply
        - {name}_metrics.txt
    
    Post-processing:
        - {name}_final.ply
"""

import sys
import subprocess
import time
from pathlib import Path


def print_banner(text):
    """Print a formatted banner"""
    print(f"\n{'='*70}")
    print(f"{text.center(70)}")
    print(f"{'='*70}\n")


def run_module(script_name, args, module_name):
    """
    Run a module script and handle errors
    
    Args:
        script_name: Name of the Python script to run
        args: List of command-line arguments
        module_name: Display name for the module
        
    Returns:
        True if successful, False otherwise
    """
    print_banner(f"RUNNING MODULE: {module_name}")
    
    print(f"Command: python3 {script_name} {' '.join(args)}")
    print(f"Starting at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    start_time = time.time()
    
    try:
        # Run the module
        result = subprocess.run(
            ["python3", script_name] + args,
            check=True,
            capture_output=False,  # Let output go to console
            text=True
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n{module_name} completed successfully!")
        print(f"Time taken: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: {module_name} failed with return code {e.returncode}")
        print(f"Time before failure: {time.time() - start_time:.1f} seconds")
        return False
    
    except KeyboardInterrupt:
        print(f"\n⚠️  {module_name} interrupted by user")
        return False
    
    except Exception as e:
        print(f"\n❌ ERROR: {module_name} failed with exception: {e}")
        return False


def check_file_exists(filepath, description):
    """
    Check if a file exists and print status
    
    Args:
        filepath: Path to check
        description: Description of the file
        
    Returns:
        True if exists, False otherwise
    """
    if Path(filepath).exists():
        size = Path(filepath).stat().st_size
        size_mb = size / (1024 * 1024)
        print(f"  ✓ {description}: {filepath} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"  ✗ {description}: {filepath} (NOT FOUND)")
        return False


def verify_outputs(base_name, stage):
    """
    Verify that expected outputs were created
    
    Args:
        base_name: Base name for output files (e.g., 'frag_1')
        stage: Which stage to verify ('preprocessing', 'algorithm', 'postprocessing')
        
    Returns:
        True if all expected files exist, False otherwise
    """
    print(f"\nVerifying {stage} outputs:")
    
    all_exist = True
    
    if stage == 'preprocessing':
        all_exist &= check_file_exists(f"{base_name}_preprocessed.ply", "Preprocessed PLY")
        all_exist &= check_file_exists(f"{base_name}_metadata.pkl", "Metadata")
    
    elif stage == 'algorithm':
        all_exist &= check_file_exists(f"{base_name}_probabilities.npy", "Probabilities")
        all_exist &= check_file_exists(f"{base_name}_features.npy", "Features")
        all_exist &= check_file_exists(f"{base_name}_model.pkl", "Trained model")
        all_exist &= check_file_exists(f"{base_name}_comparison.ply", "Comparison PLY")
        all_exist &= check_file_exists(f"{base_name}_metrics.txt", "Metrics")
    
    elif stage == 'postprocessing':
        all_exist &= check_file_exists(f"{base_name}_final.ply", "Final result")
    
    return all_exist


def main():
    """Main pipeline execution"""
    
    # Check command-line arguments
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    input_ply = sys.argv[1]
    
    # Check input file exists
    if not Path(input_ply).exists():
        print(f"❌ ERROR: Input file not found: {input_ply}")
        sys.exit(1)
    
    # Get base name for outputs
    base_name = Path(input_ply).stem
    
    # Print pipeline header
    print_banner("BREAK SURFACE DETECTION - MASTER PIPELINE")
    
    print(f"Input file: {input_ply}")
    print(f"Base name: {base_name}")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    pipeline_start = time.time()
    
    # Track success of each module
    all_successful = True
    
    # ========================================
    # MODULE 1: PREPROCESSING
    # ========================================
    
    success = run_module(
        "1_preprocessing.py",
        [input_ply],
        "MODULE 1: PREPROCESSING"
    )
    
    if not success:
        print("\n❌ Pipeline failed at preprocessing stage")
        sys.exit(1)
    
    if not verify_outputs(base_name, 'preprocessing'):
        print("\n❌ Preprocessing outputs incomplete")
        sys.exit(1)
    
    all_successful &= success
    
    # ========================================
    # MODULE 2: ALGORITHM
    # ========================================
    
    success = run_module(
        "2_algorithm.py",
        [f"{base_name}_preprocessed.ply", f"{base_name}_metadata.pkl"],
        "MODULE 2: ALGORITHM"
    )
    
    if not success:
        print("\n❌ Pipeline failed at algorithm stage")
        sys.exit(1)
    
    if not verify_outputs(base_name, 'algorithm'):
        print("\n❌ Algorithm outputs incomplete")
        sys.exit(1)
    
    all_successful &= success
    
    # ========================================
    # MODULE 3: POST-PROCESSING
    # ========================================
    
    success = run_module(
        "3_postprocessing.py",
        [f"{base_name}_probabilities.npy", f"{base_name}_metadata.pkl"],
        "MODULE 3: POST-PROCESSING"
    )
    
    if not success:
        print("\n❌ Pipeline failed at post-processing stage")
        sys.exit(1)
    
    if not verify_outputs(base_name, 'postprocessing'):
        print("\n❌ Post-processing outputs incomplete")
        sys.exit(1)
    
    all_successful &= success
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    
    pipeline_time = time.time() - pipeline_start
    
    print_banner("PIPELINE COMPLETE")
    
    if all_successful:
        print("✅ All modules completed successfully!\n")
    else:
        print("⚠️  Some modules had issues. Check logs above.\n")
    
    print("Generated outputs:")
    print("\n📁 Preprocessing:")
    print(f"   - {base_name}_preprocessed.ply")
    print(f"   - {base_name}_metadata.pkl")
    
    print("\n📁 Algorithm:")
    print(f"   - {base_name}_probabilities.npy")
    print(f"   - {base_name}_features.npy")
    print(f"   - {base_name}_model.pkl")
    print(f"   - {base_name}_comparison.ply (algorithm output vs GT)")
    print(f"   - {base_name}_metrics.txt")
    
    print("\n📁 Post-processing:")
    print(f"   - {base_name}_final.ply (final cleaned result)")
    
    print(f"\n⏱️  Total pipeline time: {pipeline_time:.1f} seconds ({pipeline_time/60:.1f} minutes)")
    print(f"🎯 Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print key results if metrics file exists
    metrics_file = f"{base_name}_metrics.txt"
    if Path(metrics_file).exists():
        print(f"\n📊 Key Results (from {metrics_file}):")
        with open(metrics_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Precision:' in line or 'Recall:' in line or 'F1 Score:' in line:
                    print(f"   {line.strip()}")
    
    print(f"\n{'='*70}")
    print("View results:")
    print(f"  Algorithm comparison: open {base_name}_comparison.ply")
    print(f"  Final result:         open {base_name}_final.ply")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)