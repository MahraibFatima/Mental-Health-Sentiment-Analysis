from src.MentalHealthAnalysis.constants import *
from src.MentalHealthAnalysis.utils.common import read_yaml, create_directories
from src.MentalHealthAnalysis.entity.config_entity import (DataIngestionConfig,
                                                DataValidationConfig,
                                                DataTransformationConfig,
                                                ModelTrainerConfig,
                                                ModelEvaluationConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        try:
            # Read YAML configuration files
            self.config = read_yaml(config_filepath)
            self.params = read_yaml(params_filepath)
            self.schema = read_yaml(schema_filepath)
            
            # Create necessary directories
            create_directories([self.config.get('artifacts_root', '')])
        except Exception as e:
            raise RuntimeError(f"Error loading configuration files: {e}")

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            config = self.config.get('data_ingestion', {})
            create_directories([config.get('root_dir', '')])

            data_ingestion_config = DataIngestionConfig(
                root_dir=config.get('root_dir', ''),
                source_URL=config.get('source_URL', ''),
                local_data_file=config.get('local_data_file', ''),
                unzip_dir=config.get('unzip_dir', '') 
            )
            return data_ingestion_config
        except Exception as e:
            raise RuntimeError(f"Error in data ingestion config: {e}")
    
    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            config = self.config.get('data_validation', {})
            schema = self.schema.get('COLUMNS', {})
            create_directories([config.get('root_dir', '')])

            data_validation_config = DataValidationConfig(
                root_dir=config.get('root_dir', ''),
                STATUS_FILE=config.get('STATUS_FILE', ''),
                unzip_data_dir=config.get('unzip_data_dir', ''),
                all_schema=schema,
            )
            return data_validation_config
        except Exception as e:
            raise RuntimeError(f"Error in data validation config: {e}")
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            config = self.config.get('data_transformation', {})
            create_directories([config.get('root_dir', '')])

            data_transformation_config = DataTransformationConfig(
                root_dir=config.get('root_dir', ''),
                data_path=config.get('data_path', ''),
            )
            return data_transformation_config
        except Exception as e:
            raise RuntimeError(f"Error in data transformation config: {e}")
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            config = self.config.get('model_trainer', {})
            params = self.params.get('xgboost_params', {})
            schema = self.schema.get('TARGET_COLUMN', {})

            create_directories([config.get('root_dir', '')])

            model_trainer_config = ModelTrainerConfig(
                root_dir=config.get('root_dir', ''),
                train_data_path=config.get('train_data_path', ''),
                test_data_path=config.get('test_data_path', ''),
                model_name=config.get('model_name', ''),
                alpha=params.get('alpha', 0.1),  # Default value if not present
                l1_ratio=params.get('l1_ratio', 0.5),  # Default value if not present
                target_column=schema.get('name', '')
            )
            return model_trainer_config
        except Exception as e:
            raise RuntimeError(f"Error in model trainer config: {e}")

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        try:
            config = self.config.get('model_evaluation', {})
            params = self.params.get('xgboost_params', {})
            schema = self.schema.get('TARGET_COLUMN', {})

            create_directories([config.get('root_dir', '')])

            model_evaluation_config = ModelEvaluationConfig(
                root_dir=config.get('root_dir', ''),
                test_data_path=config.get('test_data_path', ''),
                model_path=config.get('model_name', ''),  # Update to use model name
                all_params=params,
                metric_file_name=config.get('metric_file_name', ''),
                target_column=schema.get('name', '')
            )
            return model_evaluation_config
        except Exception as e:
            raise RuntimeError(f"Error in model evaluation config: {e}")
