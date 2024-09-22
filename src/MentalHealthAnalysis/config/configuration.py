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
            create_directories([self.config.getartifacts_root])
        except Exception as e:
            raise RuntimeError(f"Error loading configuration files: {e}")

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            config = self.config.getdata_ingestion 
            create_directories([config.root_dir])

            data_ingestion_config = DataIngestionConfig(
                root_dir=config.root_dir,
                source_URL=config.source_URL,
                local_data_file=config.local_data_file,
                unzip_dir=config.unzip_dir
            )
            return data_ingestion_config
        except Exception as e:
            raise RuntimeError(f"Error in data ingestion config: {e}")
    
    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            config = self.config.data_validation
            schema = self.schema.COLUMNS
            create_directories([config.root_dir])

            data_validation_config = DataValidationConfig(
                root_dir=config.root_dir,
                STATUS_FILE=config.STATUS_FILE,
                unzip_data_dir=config.unzip_data_dir,
                all_schema=schema,
            )
            return data_validation_config
        except Exception as e:
            raise RuntimeError(f"Error in data validation config: {e}")
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            config = self.config.data_transformation
            create_directories([config.root_dir])

            data_transformation_config = DataTransformationConfig(
                root_dir=config.root_dir,
                data_path=config.data_path,
            )
            return data_transformation_config
        except Exception as e:
            raise RuntimeError(f"Error in data transformation config: {e}")
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            config = self.config.get('model_trainer', {})
            params = self.params.get('XGBClassifier', {})
            schema = self.schema.get('TARGET_COLUMN', {})

            create_directories([config.root_dir])

            model_trainer_config = ModelTrainerConfig(
                root_dir=config.root_dir,
                train_data_path=config.train_data_path,
                test_data_path=config.test_data_path,
                model_name=config.model_name,
                learning_rate=params.get('learning_rate', 0.5),
                max_depth=params.get('max_depth', 7),
                n_estimators=params.get('n_estimators', 500),
                target_column=schema.name
            )
            return model_trainer_config
        except Exception as e:
            raise RuntimeError(f"Error in model trainer config: {e}")

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        try:
            config = self.config.model_evaluation
            params = self.params.XGBClassifier
            schema =  self.schema.TARGET_COLUMN

            create_directories([config.root_dir])

            model_evaluation_config = ModelEvaluationConfig(
                root_dir=config.root_dir,
                test_data_path=config.test_data_path,
                model_path = config.model_path,
                all_params=params,
                metric_file_name = config.metric_file_name,
                target_column = schema.name
           
            )

            return model_evaluation_config

        except Exception as e:
            raise RuntimeError(f"Error in model evaluation config: {e}")
