from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import yaml

@dataclass
class PathConfig:
    """File paths and directories"""
    data_path: str = 'data/'
    era5_path: str = 'data/full_data/era5_raw/'
    test_path: str = 'test/'
    nc_folder: str = 'results/nc/'
    labeled_nc_folder: str = 'results/nc/labeled_nc/'
    image_folder: str = 'results/img/'
    csv_folder: str = 'results/csv/'
    
    def __post_init__(self):
        # Create directories if they don't exist
        for path in [self.data_path, self.test_path, self.image_folder, self.csv_folder, self.nc_folder, self.labeled_nc_folder]:
            Path(path).mkdir(parents=True, exist_ok=True)

@dataclass 
class FileConfig:
    """Input and output file names"""
    mean_file: str = "seasonal_mean.nc"
    stdev_file: str = "seasonal_stdev.nc"
    anomaly_file: str = "seasonal_anomalies.nc"
    masked_anomaly_file: str = "seasonal_anomalies_masked.nc"
    cao_cluster_table: str = "cao_clusters.csv"
    cao_chain_table: str = "cao_chains.csv"
    ocean_mask: str = "ocean_mask.nc"
    default_config: str = "config.yaml"

@dataclass
class AlgorithmConfig:
    """Core algorithm parameters for cold air outbreak detection"""
    # Temperature anomaly threshold (standard deviations below mean)
    stdev_threshold: float = -1.5
    min_stdev_threshold: float = 3
    # Minimum cluster area in km²
    area_threshold: float = 500000
    distance_threshold_km: float = 1000
    # Time grouping parameters
    time_group: str = 'season'
    time_group_period: str = 'DJF'  # December-January-February
    
    # Variable to analyze
    temperature_var: str = "daily_t2_min"
    
    def __post_init__(self):
        """Validate algorithm parameters"""
        if self.stdev_threshold > 0:
            raise ValueError("Temperature threshold should be negative (below mean)")
        if self.area_threshold <= 0:
            raise ValueError("Area threshold must be positive")
        if self.time_group_period not in ['DJF', 'MAM', 'JJA', 'SON']:
            raise ValueError("time_group_period must be one of: DJF, MAM, JJA, SON")

@dataclass
class DatasetConfig:
    """Dataset dimension and layer names"""
    longitude_spatial_res: float = 0.25
    latitude_spatial_res: float = 0.25
    time_dim: str = 'time'
    latitude_dim_name: str = 'latitude'
    longitude_dim_name: str = 'longitude'
    mask_var: str = 'mask'
    area_var: str = 'areas'
    label_var: str = "labeled_clusters"
    label_filtered_var: str = 'labeled_clusters_filtered'
    anomaly_var: str = "anomaly"
    scaled_anomaly_var: str = "scaled_anomaly"

@dataclass
class TrackerConfig:
    """Main configuration class combining all config sections"""
    paths: PathConfig = field(default_factory=PathConfig)
    files: FileConfig = field(default_factory=FileConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'TrackerConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            paths=PathConfig(**config_dict.get('paths', {})),
            files=FileConfig(**config_dict.get('files', {})),
            algorithm=AlgorithmConfig(**config_dict.get('algorithm', {})),
            dataset=DatasetConfig(**config_dict.get('dataset', {}))
        )
    
    def to_yaml(self, config_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'paths': self.paths.__dict__,
            'files': self.files.__dict__,
            'algorithm': self.algorithm.__dict__,
            'dataset': self.dataset.__dict__
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

# Global configuration instance
_config: Optional[TrackerConfig] = None

def get_config() -> TrackerConfig:
    """Get the current global configuration"""
    global _config
    if _config is None:
        _config = TrackerConfig()
    return _config

def set_config(config: TrackerConfig):
    """Set the global configuration"""
    global _config
    _config = config

def load_config(config_path: str):
    """Load configuration from file and set as global config"""
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        config = TrackerConfig.from_yaml(config_path)
    else:
        raise ValueError("Config file must be .yaml or .yml")
    
    set_config(config)
    return config

# Convenience functions for backward compatibility
def get_paths():
    return get_config().paths

def get_files():
    return get_config().files

def get_algorithm_params():
    return get_config().algorithm

def get_dataset_params():
    return get_config().dataset

# Export commonly used parameters for easy access
def get_stdev_threshold():
    return get_config().algorithm.stdev_threshold

def get_area_threshold():
    return get_config().algorithm.area_threshold

def create_default_config_file():
    get_config().to_yaml(get_config().files.default_config)