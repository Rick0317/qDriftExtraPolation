"""
Quick script to organize existing CSV files in the workspace.

This script will:
1. Show current state of results directory
2. Preview what files would be moved from root
3. Optionally execute the cleanup

Run with: python organize_existing_results.py
"""

import pathlib
import sys

# Add src to path
sys.path.append(str(pathlib.Path(__file__).parent))

from src.utils.results_management import ResultsOrganizer


def main():
    # Initialize organizer
    root = pathlib.Path(__file__).parent
    print(f"📂 Working in directory: {root.resolve()}")
    organizer = ResultsOrganizer(root / "results")
    
    print("="*70)
    print("RESULTS ORGANIZATION TOOL")
    print("="*70)
    
    # Show current state
    print("\n📊 Current state of results directory:")
    organizer.print_summary()
    
    # Find loose CSV files in root and all subdirectories
    loose_csvs = []
    loose_jsons = []
    
    # Search recursively from workspace root
    for csv_file in root.rglob("*.csv"):
        # Skip files already in organized directories within results/
        try:
            rel_path = csv_file.relative_to(organizer.results_root)
            first_part = str(rel_path.parts[0]) if rel_path.parts else ""
            if first_part not in ['parameter_sweeps', 'extrapolation', 'benchmarks', 'archive', 'figures', 'errors']:
                loose_csvs.append(csv_file)
        except ValueError:
            # File is outside results_root (i.e., in workspace root), include it
            loose_csvs.append(csv_file)
    
    for json_file in root.rglob("*.json"):
        try:
            rel_path = json_file.relative_to(organizer.results_root)
            first_part = str(rel_path.parts[0]) if rel_path.parts else ""
            if first_part not in ['parameter_sweeps', 'extrapolation', 'benchmarks', 'archive', 'figures', 'errors']:
                loose_jsons.append(json_file)
        except ValueError:
            loose_jsons.append(json_file)
    
    print(f"\n📁 Found {len(loose_csvs)} CSV files in root directory:")
    for csv_file in sorted(loose_csvs)[:10]:  # Show first 10
        size_mb = csv_file.stat().st_size / (1024 * 1024)
        print(f"   • {csv_file.name} ({size_mb:.2f} MB)")
    
    if len(loose_csvs) > 10:
        print(f"   ... and {len(loose_csvs) - 10} more")
    
    if loose_jsons:
        print(f"\n📄 Found {len(loose_jsons)} JSON files in root directory")
    
    # Preview cleanup
    if loose_csvs or loose_jsons:
        print("\n" + "="*70)
        print("PREVIEW: Where files would be moved")
        print("="*70)
        
        # Preview workspace root CSVs
        for csv_file in loose_csvs:
            filename = csv_file.name
            sweep_type = organizer._infer_sweep_type_from_csv(csv_file) if csv_file.suffix == '.csv' else None
            
            if 'parameter_sweep' in filename.lower() or 'qdrift_qpe' in filename.lower():
                if sweep_type == 'chebyshev':
                    category = 'chebyshev_interpolation'
                elif sweep_type == 'full':
                    category = 'full_sweeps'
                else:
                    category = 'parameter_sweeps'
            elif 'extrapolation' in filename.lower():
                category = 'extrapolation'
            elif 'benchmark' in filename.lower():
                category = 'benchmarks'
            else:
                if sweep_type == 'chebyshev':
                    category = 'chebyshev_interpolation'
                elif sweep_type == 'full':
                    category = 'full_sweeps'
                else:
                    category = 'parameter_sweeps'
            
            from datetime import datetime
            mod_time = datetime.fromtimestamp(csv_file.stat().st_mtime)
            month = mod_time.strftime("%Y-%m")
            target = f"results/parameter_sweeps/{category}/{month}/{filename}"
            print(f"Would move: {csv_file.relative_to(root)} -> {target}")
        
        # Ask user if they want to proceed
        print("\n" + "="*70)
        response = input("\n❓ Execute cleanup? This will move files. (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            print("\n✅ Executing cleanup...")
            
            # Move each file manually to ensure all files get organized
            moved_count = 0
            import shutil
            from datetime import datetime
            
            # Track where each file was moved (stem -> target_dir)
            csv_target_dirs = {}
            
            # First, determine target directories and move CSVs
            for csv_file in loose_csvs:
                filename = csv_file.name
                sweep_type = organizer._infer_sweep_type_from_csv(csv_file) if csv_file.suffix == '.csv' else None
                
                # Determine target directory
                if 'parameter_sweep' in filename.lower() or 'qdrift_qpe' in filename.lower():
                    if sweep_type == 'chebyshev':
                        target_dir = organizer.dirs['chebyshev_interpolation']
                    elif sweep_type == 'full':
                        target_dir = organizer.dirs['full_sweeps']
                    else:
                        target_dir = organizer.dirs['parameter_sweeps']
                elif 'extrapolation' in filename.lower():
                    target_dir = organizer.dirs['extrapolation']
                elif 'benchmark' in filename.lower():
                    target_dir = organizer.dirs['benchmarks']
                else:
                    if sweep_type == 'chebyshev':
                        target_dir = organizer.dirs['chebyshev_interpolation']
                    elif sweep_type == 'full':
                        target_dir = organizer.dirs['full_sweeps']
                    else:
                        target_dir = organizer.dirs['parameter_sweeps']
                
                # Create month subdirectory
                mod_time = datetime.fromtimestamp(csv_file.stat().st_mtime)
                month_dir = target_dir / mod_time.strftime("%Y-%m")
                month_dir.mkdir(parents=True, exist_ok=True)
                
                # Track target directory for matching JSON files
                csv_target_dirs[csv_file.stem] = month_dir
                
                target_path = month_dir / filename
                shutil.move(str(csv_file), str(target_path))
                print(f"Moved: {csv_file.relative_to(root)} -> {target_path.relative_to(root)}")
                moved_count += 1
            
            # Move JSON files to same location as their corresponding CSV
            for json_file in loose_jsons:
                # Check if we have a matching CSV that was moved
                if json_file.stem in csv_target_dirs:
                    month_dir = csv_target_dirs[json_file.stem]
                    target_path = month_dir / json_file.name
                    shutil.move(str(json_file), str(target_path))
                    print(f"Moved: {json_file.relative_to(root)} -> {target_path.relative_to(root)}")
                    moved_count += 1
                else:
                    # JSON without matching CSV - use filename heuristics
                    filename = json_file.name
                    if 'parameter_sweep' in filename.lower() or 'qdrift_qpe' in filename.lower():
                        target_dir = organizer.dirs['parameter_sweeps']
                    elif 'extrapolation' in filename.lower():
                        target_dir = organizer.dirs['extrapolation']
                    elif 'benchmark' in filename.lower():
                        target_dir = organizer.dirs['benchmarks']
                    else:
                        target_dir = organizer.dirs['parameter_sweeps']
                    
                    mod_time = datetime.fromtimestamp(json_file.stat().st_mtime)
                    month_dir = target_dir / mod_time.strftime("%Y-%m")
                    month_dir.mkdir(parents=True, exist_ok=True)
                    target_path = month_dir / json_file.name
                    shutil.move(str(json_file), str(target_path))
                    print(f"Moved: {json_file.relative_to(root)} -> {target_path.relative_to(root)}")
                    moved_count += 1
            
            print(f"\n✅ Cleanup complete! Moved {moved_count} files.")
            print("\n📊 New state of results directory:")
            organizer.print_summary()
        else:
            print("\n❌ Cleanup cancelled. No files were moved.")
    else:
        print("\n✅ No loose CSV or JSON files found in root directory!")
    
    # Optional: Show archive preview
    print("\n" + "="*70)
    print("ARCHIVE PREVIEW: Files older than 60 days")
    print("="*70)
    old_files = organizer.archive_old_results(days_old=60, dry_run=True)
    
    if old_files:
        print(f"\n📦 Found {len(old_files)} files that could be archived")
        response = input("\n❓ Archive old files? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            organizer.archive_old_results(days_old=60, dry_run=False)
            print("\n✅ Archiving complete!")
    else:
        print("\n✅ No old files to archive")
    
    print("\n" + "="*70)
    print("Done! Your results are now organized.")
    print("="*70)
    
    # Show final summary
    print("\n📋 Final summary:")
    organizer.print_summary()
    
    print("\n💡 Tips:")
    print("   • Use ResultsOrganizer.save_dataframe() for new results")
    print("   • Check REFACTORING_GUIDE.md for detailed documentation")
    print("   • See results/interp/extrapolation.ipynb for examples")


if __name__ == "__main__":
    main()
