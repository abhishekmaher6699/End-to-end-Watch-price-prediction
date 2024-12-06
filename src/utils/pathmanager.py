import os

class PathManager:
    
    def __init__(self):
        self.project_root = self._get_project_root()
        self.artifacts_dir = os.path.join(self.project_root, 'artifacts')
        self.data_raw_dir = os.path.join(self.artifacts_dir, 'data', 'raw')
        self.data_processed_dir = os.path.join(self.artifacts_dir, 'data', 'processed')
        self.models_dir = os.path.join(self.artifacts_dir, 'models')
        self.reports_dir = os.path.join(self.project_root, 'reports')

        os.makedirs(self.data_raw_dir, exist_ok=True)
        os.makedirs(self.data_processed_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
    def _get_project_root(self) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    def get_raw_data_path(self) -> str:
        return os.path.join(self.data_raw_dir, "raw_data.csv")
    
    def get_processed_data_path(self) -> str:
        return os.path.join(self.data_processed_dir, 'processed_data.csv')

    def get_train_data_path(self) -> str:
        return os.path.join(self.data_processed_dir, 'train.csv')
    
    def get_test_data_path(self) -> str:
        return os.path.join(self.data_processed_dir, 'test.csv')
    
    def get_params_path(self) -> str:
        return os.path.join(self.project_root, 'params.yaml')
        
    def get_pipeline_path(self) -> str:
        return os.path.join(self.models_dir, 'pipeline.pkl')
    
    def get_columns_path(self) -> str:
        return os.path.join(self.models_dir, 'columns.pkl')
    
    def get_reports_path(self) -> str:
        return os.path.join(self.reports_dir, 'evaluation.txt')

    def get_chart_path(self) -> str:
        return os.path.join(self.reports_dir, 'chart.png')
