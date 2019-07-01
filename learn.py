import hashlib
import pickle
import random
from io import BytesIO
from typing import Dict, Tuple

import numpy as np
import sklearn.metrics
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


def get_dataset(seed: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    global BOSTON
    if 'BOSTON' not in globals():
        BOSTON = load_boston()
    boston_train_data, boston_test_data, boston_train_target, boston_test_target = \
        train_test_split(BOSTON.data, BOSTON.target, test_size=0.1, random_state=seed)

    def boston_dataset_factory(data: np.ndarray, target: np.ndarray) -> Dict[str, np.ndarray]:
        #  Asserts are acceptable, it's not a production server, just learning script
        assert isinstance(data, np.ndarray)
        assert isinstance(target, np.ndarray)
        return {
            'data': data,
            'target': target,
            'feature_names': BOSTON['feature_names']
        }

    return boston_dataset_factory(boston_train_data, boston_train_target), \
        boston_dataset_factory(boston_test_data, boston_test_target)


def calculate_metrics(model: KNeighborsRegressor, test_dataset: Dict[str, np.ndarray]) -> Dict[str, float]:
    prediction = model.predict(test_dataset['data'])
    args = (test_dataset['target'], prediction)
    return {
        'MAE': sklearn.metrics.mean_absolute_error(*args),
        'MSE': sklearn.metrics.mean_squared_error(*args),
        'R2': sklearn.metrics.r2_score(*args)
    }


def learn(k: int, dataset: Dict[str, np.ndarray]) -> KNeighborsRegressor:
    regressor = KNeighborsRegressor(k, 'distance')
    model = regressor.fit(dataset['data'], dataset['target'])
    return model


def save_model(model: KNeighborsRegressor, metrics: Dict[str, float], neighbors_count: int, seed: int) -> None:
    serialized_model = BytesIO()
    pickle.dump(model, serialized_model)
    md5 = hashlib.md5()
    md5.update(serialized_model.read())
    md5_string = md5.hexdigest()
    yaml_metrics = {k: float(v) for k, v in metrics.items()}
    # md5 is important to protect server from malicious pickle
    yaml_metrics["serialized_model_md5"] = md5_string
    yaml_metrics["neghbors_count"] = neighbors_count
    yaml_metrics["random_seed"] = seed
    with open('best_model.pckl', 'wb') as dmp_file:
        dmp_file.write(serialized_model.read())
    with open('best_model_info.yaml', 'w') as md5_file:
        import yaml
        yaml.safe_dump(yaml_metrics, md5_file)


def main():
    seed = random.randint(0, 2 ** 32)
    boston_train, boston_test = get_dataset(seed)
    best_mse = 2 ** 63
    best_metrics = None
    best_model = None
    best_neighbors_count = None

    # Searching best neighbors count to minimize MSE
    for k in range(1, boston_train['target'].shape[0]):
        model = learn(k, boston_train)
        metrics = calculate_metrics(model, boston_test)
        if metrics['MSE'] < best_mse:
            best_mse = metrics['MSE']
            best_metrics = metrics
            best_model = model
            best_neighbors_count = k

    save_model(best_model, best_metrics, best_neighbors_count, seed)


if __name__ == '__main__':
    main()
