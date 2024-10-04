from metaflow import FlowSpec, Parameter, step


class EPLClassifierTrain(FlowSpec):
    """Metaflow Flow to process data and train model to
    classify outcomes of English Premier League matches."""
    data_path = Parameter('data_path', default='../data/final_dataset.csv',
                          type=str, required=False)
    standings = Parameter('standings_path', default='../data/EPLStandings.csv',
                          type=str, required=False)
    feature_pct = Parameter('feature_pct', default=50, type=int,
                            required=False)
    num_splits = Parameter('cv_splits', default=5,
                           type=int, required=False)
    random_state = Parameter('random_seed', default=42,
                             type=int, required=False)
    mod_name = Parameter('registered_model_name', type=str,
                         default='best_metaflow_model', required=False)
    exp_name = Parameter('experiment', type=str, required=True)

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
        import epl_processing as ep
        self.train_data, self.test_data, self.standings_df = \
            ep.process_data(epl_data=self.data_df,
                            standings_data=self.standings_df,
                            feature_percentile=self.feature_pct)

        self.train_y = self.train_data['Result']
        self.train_data = self.train_data.drop(columns=['Result'])

        print('Data Processing Complete.')
        self.next(self.model_training)

    @step
    def model_training(self):
        import model_training as mtrain
        import mlflow
        # perform mlflow training using hyperopt
        mlflow.set_tracking_uri('http://127.0.0.1:8080')
        mlflow.set_experiment(self.exp_name)
        self.best_run, self.best_id = mtrain.hp_tuning(
            train_data=self.train_data,
            train_y=self.train_y, num_splits=self.num_splits,
            random_state=self.random_state)

        print('Model Training Complete.')
        self.next(self.model_logging)

    @step
    def model_logging(self):
        import mlflow
        mlflow.set_experiment(self.exp_name)
        model_path = f'runs:/{self.best_id}/artifacts/better_models'
        mlflow.register_model(model_uri=model_path,
                              name=self.mod_name)
        print('Model successfully logged.')
        self.next(self.end)

    @step
    def end(self):
        print('Training Flow complete')
        print(f"Best model type: {self.best_run['classifier_type']}")
        print(f'Best model id: {self.best_id}')


if __name__ == '__main__':
    EPLClassifierTrain()
