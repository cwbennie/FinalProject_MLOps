from metaflow import FlowSpec, Parameter, step, JSONType
from metaflow import kubernetes, conda_base, retry, timeout


@conda_base(libraries={'hyperopt': '0.2.7', 'mlflow': '2.5.0',
                       'numpy': '1.23.5', 'pandas': '1.5.3',
                       'scikit-learn': '1.1.3', 'xgboost': '1.6.1',
                       'databricks-cli': '0.17.6'},
                       python='3.10')
class EPLKubernetesTrain(FlowSpec):
    """Metaflow Flow to process data and train model to
    classify outcomes of English Premier League matches."""
    data_path = Parameter('data_path', default='../data/final_dataset.csv',
                          type=str, required=False)
    standings = Parameter('standings_path', default='../data/EPLStandings.csv',
                          type=str, required=False)
    feature_pct = Parameter('feature_pct', default=75, type=int,
                            required=False)
    num_splits = Parameter('cv_splits', default=5,
                           type=int, required=False)
    random_state = Parameter('random_seed', default=42,
                             type=int, required=False)
    mod_name = Parameter('registered_model_name', type=str,
                         default='best_kube_model', required=False)
    exp_name = Parameter('experiment', type=str, required=True,
                         default='kubernetes_training')
    scoring = Parameter('scoring', type=JSONType,
                        default={'accuracy': 'accuracy',
                                 'f1_macro': 'f1_macro'})
    grid_search = Parameter('grid_search', type=bool, default=False)
    mlflow_uri = Parameter('mlflow_uri', type=str, required=True,
                           default='https://mlops-680-370980073666.us-west2.run.app')

    @step
    def start(self):
        """Start step: Loads the data for processing/training."""
        import pandas as pd
        self.data_df = pd.read_csv(self.data_path, index_col=0)
        self.standings_df = pd.read_csv(self.standings, index_col=0)

        print("Data loaded successfully")
        self.next(self.process_data)

    @step
    def process_data(self):
        """Step to process EPL training data to be used in training"""
        import utils.epl_processing as ep
        self.train_data, self.test_data, self.standings_df = \
            ep.process_data(epl_data=self.data_df,
                            standings_data=self.standings_df,
                            feature_percentile=self.feature_pct)

        self.train_y = self.train_data['Result']
        self.train_data = self.train_data.drop(columns=['Result'])

        print('Data Processing Complete.')
        self.next(self.prepare_grids)

    @step
    def prepare_grids(self):
        import utils.model_training as mtrain
        self.param_grids = mtrain.get_param_grids(self.random_state)

        self.next(self.model_training, foreach='param_grids')

    @kubernetes(cpu=4, memory=24000, disk=24000)
    @retry(times=3)
    @timeout(minutes=5)
    @step
    def model_training(self):
        import utils.model_training as mtrain
        import mlflow
        # perform mlflow training using sklearn grid or random search
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.exp_name)

        self.model_info = mtrain.search_cv(train_data=self.train_data,
                                           train_y=self.train_y,
                                           clf_info=self.input,
                                           scoring=self.scoring,
                                           num_splits=self.num_splits,
                                           random_state=self.random_state,
                                           grid=self.grid_search)

        print('Model Training Complete.')
        self.next(self.model_logging)

    @step
    def model_logging(self, inputs):
        import mlflow
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.exp_name)

        # get the best result from the training step
        def score_mods(inps):
            mod_dicts = [mod.model_info for mod in inps]
            mod_dicts = sorted(mod_dicts,
                               key=lambda x: (x['f1_score'],
                                              x['accuracy']))
            return mod_dicts

        self.train_results = score_mods(inputs)
        self.best_model = self.train_results[0]
        self.best_id = self.best_model['run_id']
        model_path = f'runs:/{self.best_id}/artifacts/better_models'
        mlflow.register_model(model_uri=model_path,
                              name=self.mod_name)
        print('Model successfully logged.')
        self.next(self.end)

    @step
    def end(self):
        print('Training Flow complete')
        print(f"Best model type: {self.best_model['type']}")
        print(f'Best model id: {self.best_id}')


if __name__ == '__main__':
    EPLKubernetesTrain()
