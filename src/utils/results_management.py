"""
Results management utilities for organizing simulation outputs.

This module provides tools for:
- Organizing CSV files with proper directory structure
- Auto-generating filenames with metadata
- Moving/archiving old results
- Searching and filtering result files
"""

import pathlib
import shutil
import json
import pandas as pd
from typing import Dict, List, Optional, Literal
from datetime import datetime
import re


class ResultsOrganizer:
    """
    Manages organization of simulation result files.
    
    Recommended directory structure:
    results/
    ├── parameter_sweeps/
    │   ├── YYYY-MM/
    │   │   ├── hamiltonian_name/
    │   │   │   ├── sweep_YYYY-MM-DD_HH-MM.csv
    │   │   │   └── sweep_YYYY-MM-DD_HH-MM.json
    ├── extrapolation/
    │   ├── YYYY-MM/
    │   │   └── ...
    ├── benchmarks/
    │   └── ...
    └── archive/
        └── ...
    """
    
    def __init__(self, results_root: pathlib.Path):
        """
        Initialize results organizer.
        
        Args:
            results_root: Root directory for all results (e.g., 'results/')
        """
        self.results_root = pathlib.Path(results_root)
        self.results_root.mkdir(parents=True, exist_ok=True)
        
        # Create standard subdirectories
        self.dirs = {
            'parameter_sweeps': self.results_root / 'parameter_sweeps',
            'chebyshev_interpolation': self.results_root / 'parameter_sweeps' / 'chebyshev_interpolation',
            'full_sweeps': self.results_root / 'parameter_sweeps' / 'full_sweeps',
            'extrapolation': self.results_root / 'extrapolation',
            'benchmarks': self.results_root / 'benchmarks',
            'archive': self.results_root / 'archive',
            'figures': self.results_root / 'figures',
            'errors': self.results_root / 'errors'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _infer_sweep_type_from_csv(self, csv_path: pathlib.Path) -> Literal['chebyshev', 'full', 'unknown']:
        """
        Infer whether a CSV is a Chebyshev interpolation (<10 time values) or full sweep.
        
        Args:
            csv_path: Path to CSV file
        
        Returns:
            'chebyshev' if < 10 unique time values, 'full' otherwise, 'unknown' on error
        """
        try:
            df = pd.read_csv(csv_path)
            if 'time' in df.columns:
                unique_times = df['time'].nunique()
                return 'chebyshev' if unique_times < 10 else 'full'
        except Exception:
            pass
        return 'unknown'
    
    def generate_filename(
        self,
        experiment_type: Literal['parameter_sweep', 'extrapolation', 'benchmark'],
        metadata: Optional[Dict] = None,
        extension: str = 'csv',
        include_timestamp: bool = True
    ) -> str:
        """
        Generate a standardized filename with metadata.
        
        Args:
            experiment_type: Type of experiment
            metadata: Optional metadata dict (e.g., {'hamiltonian': 'H2', 'num_ancilla': 10})
            extension: File extension without dot
            include_timestamp: Include timestamp in filename
        
        Returns:
            Generated filename
        
        Example:
            >>> org.generate_filename('parameter_sweep', 
            ...                       {'hamiltonian': 'H2', 'ancilla': 10})
            'parameter_sweep_H2_ancilla10_2025-12-19_14-30-15.csv'
        """
        parts = [experiment_type]
        
        if metadata:
            for key, value in sorted(metadata.items()):
                # Sanitize the value for filename
                clean_value = str(value).replace(' ', '_').replace('/', '-')
                parts.append(f"{key}_{clean_value}")
        
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            parts.append(timestamp)
        
        filename = '_'.join(parts) + f'.{extension}'
        return filename
    
    def get_save_path(
        self,
        experiment_type: Literal['parameter_sweep', 'extrapolation', 'benchmark'],
        filename: Optional[str] = None,
        metadata: Optional[Dict] = None,
        organize_by_month: bool = True,
        organize_by_hamiltonian: bool = False,
        num_time_values: Optional[int] = None
    ) -> pathlib.Path:
        """
        Get the full path where a result file should be saved.
        
        Args:
            experiment_type: Type of experiment
            filename: Optional filename (will be auto-generated if None)
            metadata: Metadata for filename generation and organization
            organize_by_month: Create YYYY-MM subdirectory
            organize_by_hamiltonian: Create hamiltonian subdirectory
            num_time_values: Number of time values in sweep (for auto-categorization)
        
        Returns:
            Full path for saving the file
        """
        # Map experiment type to base directory with sweep type distinction
        if experiment_type == 'parameter_sweep':
            if num_time_values is not None and num_time_values < 10:
                base_dir = self.dirs['chebyshev_interpolation']
            elif num_time_values is not None:
                base_dir = self.dirs['full_sweeps']
            else:
                # Default to parameter_sweeps root if unknown
                base_dir = self.dirs['parameter_sweeps']
        elif experiment_type == 'extrapolation':
            base_dir = self.dirs['extrapolation']
        elif experiment_type == 'benchmark':
            base_dir = self.dirs['benchmarks']
        else:
            base_dir = self.dirs['parameter_sweeps']
        
        # Add month subdirectory if requested
        if organize_by_month:
            month_dir = datetime.now().strftime("%Y-%m")
            base_dir = base_dir / month_dir
        
        # Add hamiltonian subdirectory if requested
        if organize_by_hamiltonian and metadata and 'hamiltonian' in metadata:
            ham_name = metadata['hamiltonian'].replace(' ', '_')
            base_dir = base_dir / ham_name
        
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            filename = self.generate_filename(experiment_type, metadata)
        
        return base_dir / filename
    
    def save_dataframe(
        self,
        df: pd.DataFrame,
        experiment_type: str,
        metadata: Optional[Dict] = None,
        save_json: bool = True,
        organize_by_month: bool = True,
        organize_by_hamiltonian: bool = False
    ) -> pathlib.Path:
        """
        Save a DataFrame with organized directory structure.
        
        Args:
            df: DataFrame to save
            experiment_type: Type of experiment
            metadata: Metadata for organization and filename
            save_json: Also save metadata as JSON file
            organize_by_month: Organize by month
            organize_by_hamiltonian: Organize by hamiltonian name
        
        Returns:
            Path where CSV was saved
        """
        csv_path = self.get_save_path(
            experiment_type,
            metadata=metadata,
            organize_by_month=organize_by_month,
            organize_by_hamiltonian=organize_by_hamiltonian
        )
        
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved CSV: {csv_path}")
        
        # Save metadata as JSON
        if save_json and metadata:
            json_path = csv_path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            print(f"✓ Saved metadata: {json_path}")
        
        return csv_path
    
    def find_results(
        self,
        experiment_type: Optional[str] = None,
        pattern: Optional[str] = None,
        metadata_filter: Optional[Dict] = None,
        extension: str = 'csv'
    ) -> List[pathlib.Path]:
        """
        Find result files matching criteria.
        
        Args:
            experiment_type: Filter by experiment type
            pattern: Glob pattern to match (e.g., '*H2*')
            metadata_filter: Filter by metadata values
            extension: File extension to search for
        
        Returns:
            List of matching file paths
        """
        if experiment_type:
            type_mapping = {
                'parameter_sweep': 'parameter_sweeps',
                'extrapolation': 'extrapolation',
                'benchmark': 'benchmarks'
            }
            search_dir = self.dirs[type_mapping.get(experiment_type, experiment_type)]
        else:
            search_dir = self.results_root
        
        # Build glob pattern
        if pattern:
            glob_pattern = f"**/{pattern}.{extension}"
        else:
            glob_pattern = f"**/*.{extension}"
        
        matching_files = list(search_dir.glob(glob_pattern))
        
        # Filter by metadata if provided
        if metadata_filter:
            filtered_files = []
            for file_path in matching_files:
                # Check if filename contains metadata keys/values
                filename = file_path.stem
                match = True
                for key, value in metadata_filter.items():
                    pattern = f"{key}[_-]{value}"
                    if not re.search(pattern, filename, re.IGNORECASE):
                        match = False
                        break
                if match:
                    filtered_files.append(file_path)
            matching_files = filtered_files
        
        return sorted(matching_files)
    
    def archive_old_results(
        self,
        days_old: int = 30,
        experiment_type: Optional[str] = None,
        dry_run: bool = True
    ) -> List[pathlib.Path]:
        """
        Archive result files older than specified days.
        
        Args:
            days_old: Archive files older than this many days
            experiment_type: Only archive specific experiment type
            dry_run: If True, only print what would be archived
        
        Returns:
            List of archived (or would-be archived) files
        """
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        archived_files = []
        
        if experiment_type:
            type_mapping = {
                'parameter_sweep': 'parameter_sweeps',
                'extrapolation': 'extrapolation',
                'benchmark': 'benchmarks'
            }
            search_dirs = [self.dirs[type_mapping.get(experiment_type, experiment_type)]]
        else:
            search_dirs = [self.dirs['parameter_sweeps'], 
                          self.dirs['extrapolation'],
                          self.dirs['benchmarks']]
        
        for search_dir in search_dirs:
            for file_path in search_dir.rglob('*.csv'):
                # Get file modification time
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if mod_time < cutoff_date:
                    # Determine archive path (preserve directory structure)
                    rel_path = file_path.relative_to(self.results_root)
                    archive_path = self.dirs['archive'] / rel_path
                    
                    if dry_run:
                        print(f"Would archive: {file_path} -> {archive_path}")
                    else:
                        archive_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(file_path), str(archive_path))
                        print(f"Archived: {file_path} -> {archive_path}")
                    
                    archived_files.append(file_path)
        
        return archived_files
    
    def cleanup_root_directory(
        self,
        pattern: str = "*.csv",
        dry_run: bool = True,
        recursive: bool = True
    ) -> List[pathlib.Path]:
        """
        Move loose CSV files from root to appropriate directories.
        
        This helps clean up files that weren't saved in the organized structure.
        Searches recursively through all subdirectories to find scattered files.
        
        Args:
            pattern: File pattern to match
            dry_run: If True, only show what would be moved
            recursive: If True, search recursively through all subdirectories
        
        Returns:
            List of moved (or would-be moved) files
        """
        moved_files = []
        
        # Find all files matching pattern recursively from root
        search_pattern = f"**/{pattern}" if recursive else pattern
        for file_path in self.results_root.glob(search_pattern):
            if not file_path.is_file():
                continue
            
            # Skip files already in organized directories
            try:
                rel_path = file_path.relative_to(self.results_root)
                # Skip if already in one of our organized directories
                first_part = str(rel_path.parts[0]) if rel_path.parts else ""
                if first_part in ['parameter_sweeps', 'extrapolation', 'benchmarks', 'archive', 'figures', 'errors']:
                    continue
            except ValueError:
                continue
            
            filename = file_path.name
            
            # Determine experiment type and sweep category from filename
            if 'parameter_sweep' in filename.lower() or 'qdrift_qpe' in filename.lower():
                # Check if it's a Chebyshev interpolation or full sweep
                sweep_type = self._infer_sweep_type_from_csv(file_path)
                if sweep_type == 'chebyshev':
                    target_dir = self.dirs['chebyshev_interpolation']
                elif sweep_type == 'full':
                    target_dir = self.dirs['full_sweeps']
                else:
                    target_dir = self.dirs['parameter_sweeps']
            elif 'extrapolation' in filename.lower() or 'extrap' in filename.lower():
                target_dir = self.dirs['extrapolation']
            elif 'benchmark' in filename.lower():
                target_dir = self.dirs['benchmarks']
            else:
                # Try to infer from content
                sweep_type = self._infer_sweep_type_from_csv(file_path)
                if sweep_type == 'chebyshev':
                    target_dir = self.dirs['chebyshev_interpolation']
                elif sweep_type == 'full':
                    target_dir = self.dirs['full_sweeps']
                else:
                    target_dir = self.dirs['parameter_sweeps']
            
            # Create month subdirectory based on file modification time
            mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            month_dir = target_dir / mod_time.strftime("%Y-%m")
            month_dir.mkdir(parents=True, exist_ok=True)
            
            target_path = month_dir / filename
            
            if dry_run:
                print(f"Would move: {file_path.relative_to(self.results_root.parent)} -> {target_path.relative_to(self.results_root)}")
            else:
                shutil.move(str(file_path), str(target_path))
                print(f"Moved: {file_path.relative_to(self.results_root.parent)} -> {target_path.relative_to(self.results_root)}")
            
            moved_files.append(file_path)
        
        return moved_files
    
    def get_summary(self) -> Dict[str, int]:
        """
        Get summary statistics about stored results.
        
        Returns:
            Dictionary with counts of files in each category
        """
        summary = {}
        
        for name, dir_path in self.dirs.items():
            csv_count = len(list(dir_path.rglob('*.csv')))
            json_count = len(list(dir_path.rglob('*.json')))
            summary[f"{name}_csv"] = csv_count
            summary[f"{name}_json"] = json_count
        
        return summary
    
    def print_summary(self) -> None:
        """Print a formatted summary of stored results."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("Results Directory Summary")
        print("="*60)
        
        for key, count in summary.items():
            category = key.rsplit('_', 1)[0]
            file_type = key.rsplit('_', 1)[1].upper()
            print(f"{category:20s} {file_type:5s}: {count:5d} files")
        
        print("="*60 + "\n")


def load_results_with_metadata(
    csv_path: pathlib.Path,
    auto_load_json: bool = True
) -> tuple[pd.DataFrame, Optional[Dict]]:
    """
    Load CSV results and associated JSON metadata.
    
    Args:
        csv_path: Path to CSV file
        auto_load_json: Automatically load companion JSON file if it exists
    
    Returns:
        Tuple of (DataFrame, metadata_dict or None)
    """
    df = pd.read_csv(csv_path)
    
    metadata = None
    if auto_load_json:
        json_path = csv_path.with_suffix('.json')
        if json_path.exists():
            with open(json_path, 'r') as f:
                metadata = json.load(f)
    
    return df, metadata


# Example usage patterns
if __name__ == "__main__":
    # Example 1: Initialize organizer
    organizer = ResultsOrganizer(pathlib.Path("results"))
    
    # Example 2: Generate organized save path
    save_path = organizer.get_save_path(
        experiment_type='parameter_sweep',
        metadata={
            'hamiltonian': 'H2_minimal',
            'num_ancilla': 10,
            'alpha': 0.5
        },
        organize_by_month=True,
        organize_by_hamiltonian=True
    )
    print(f"Would save to: {save_path}")
    
    # Example 3: Clean up root directory (dry run)
    organizer.cleanup_root_directory(dry_run=True)
    
    # Example 4: Print summary
    organizer.print_summary()
