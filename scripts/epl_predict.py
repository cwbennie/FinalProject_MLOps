from metaflow import FlowSpec, Parameter, step, Flow


class EPLPredict(FlowSpec):
    """Metaflow Flow to process data and train model to
    classify outcomes of English Premier League matches."""
    output_path = Parameter('output', type=str,
                            default='../data/metaflow_predictions.csv')
    partial = Parameter('partial_inference', type=bool,
                        default=False)
    test_size = Parameter('test_size', type=float, default=0.2)

    @step
    def start(self):
        """Start step: Load the test data from the training run."""
        train_run = Flow('EPLClassifierTrain').latest_run
        self.test_data = train_run['process_data'].task.data.test_data

        self.test_y = self.test_data['Result']
        self.test_data.drop(columns=['Result'], inplace=True)

        print("Data loaded successfully")
        self.next(self.load_model)

    @step
    def load_model(self):
        import mlflow
        train_run = Flow('EPLClassifierTrain').latest_run
        self.best_id = train_run['model_training'].task.data.best_id
        self.exp_name = train_run['model_training'].task.data.exp_name
        mlflow.set_tracking_uri('http://127.0.0.1:8080')
        mlflow.set_experiment(self.exp_name)

        self.model_uri = f'runs:/{self.best_id}/better_models'

        self.logged_model = mlflow.sklearn.load_model(self.model_uri)

        print('Model Loaded.')
        self.next(self.test_predict)

    @step
    def test_predict(self):
        from model_inference import predict

        self.predictions, self.accuracy, self.mod_f1 = predict(
            model=self.logged_model, test_data=self.test_data,
            test_y=self.test_y, partial=self.partial, test_size=self.test_size)

        print('Test Predictions Completed.')
        print(f'Test accuracy: {self.accuracy}')
        print(f'Test F1: {self.f1_score}')
        self.next(self.save_predictions)

    @step
    def save_predictions(self):
        self.test_set['Result'] = self.test_y
        self.test_set['Predictions'] = self.predictions

        self.test_set.to_csv(self.output_path)

        print('Predictions logged.')
        self.next(self.end)

    @step
    def end(self):
        print('Prediction Flow complete')
        print(f'Output location: {self.output_path}')


if __name__ == '__main__':
    EPLPredict()
