from dataclasses import dataclass
from pathlib import Path

# from ->  research/01_data_ingestion.ipynb __cell-5__
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

# from -> research/02_data_validation.ipynb __cell-5__
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict

# from -> research/03_data_transformation.ipynb __cell-5__
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path

# from -> research/04_model_trainer.ipynb __cell-5__
@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    learning_rate: float
    max_depth: int
    n_estimators: int
    target_column: str

#from -> research/05_model_evaluation.ipynb
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str