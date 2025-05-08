import os
import logging
import logging.handlers

import numpy as np
import pandas as pd
from pathlib import Path
from fedot import Fedot
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score as f1,
    accuracy_score as accuracy,
    mean_absolute_error as mae,
    mean_squared_error as mse,
    r2_score as r2
)

SEED = 42
np.random.seed(SEED)


def setup_logger(
        logger_name: str,
        log_file: str,
        level=logging.INFO,
        log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        rotation_bytes: int = 5_242_880,  # 5MB
        backup_count: int = 3
) -> logging.Logger:
    log_dir = os.path.dirname(log_file)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(log_format)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=rotation_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def run_experiment(target: str, task: str, data_path: str, save_name: str, time=10):
    base_path = r"logs"
    path_to_save = os.path.join(base_path, save_name)
    os.makedirs(path_to_save, exist_ok=True)
    log_file = os.path.join(path_to_save, f"experiment_{save_name}.log")

    logger = setup_logger(logger_name='MADD', log_file=log_file, level=logging.DEBUG)

    logger.info(f"Dataset: {save_name}")
    logger.info(f"Target: {target}")
    logger.info(f"Task: {task}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Save path: {path_to_save}")
    logger.info("Disbalance of classes:")

    df_ds = pd.read_csv(data_path)
    logger.info(f"{df_ds[target].value_counts(normalize=True)}")

    logger.info('-' * 50)


    X, y = df_ds.drop(columns=target), df_ds[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED)

    model = Fedot(
        problem=task,
        seed=SEED,
        timeout=time,
        n_jobs=-1,
        with_tuning=True,
        initial_assumption=PipelineBuilder() \
        .add_node('scaling')
        .add_branch('catboost', 'xgboost', 'lgbm')
        .join_branches('blending').build()
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    logger.info(f'Model graph description:\n{model.current_pipeline.graph_description}')
    model.current_pipeline.show(save_path=os.path.join(path_to_save, f'{save_name}_pipeline.png'))

    if task == 'classification':
        logger.info(f"Accuracy score: {accuracy(y_test, y_pred)}")
        logger.info(f"F-1 score: {f1(y_test, y_pred)}")
    else:
        logger.info(f"MAE score: {mae(y_test, y_pred)}")
        logger.info(f"MSE score: {mse(y_test, y_pred)}")
        logger.info(f"R2 score: {r2(y_test, y_pred)}")

    pipeline = model.current_pipeline

    pipeline.save(path=path_to_save, create_subdir=True, is_datetime_in_path=False)
    logger.info(f'Pipeline saved to {path_to_save}\n')
    logger.info('='*50)
    logger.info('\n')


if __name__ == '__main__':
    data_1024 = [
        r"C:\Users\user\Desktop\madd_automl\data\1024_data\alz_1024.csv",
        r"C:\Users\user\Desktop\madd_automl\data\1024_data\cancer_clear_1024.csv",
        r"C:\Users\user\Desktop\madd_automl\data\1024_data\dislip_1024.csv",
        r"C:\Users\user\Desktop\madd_automl\data\1024_data\parkinson_1024.csv",
        r"C:\Users\user\Desktop\madd_automl\data\1024_data\resistance_1024.csv",
        r"C:\Users\user\Desktop\madd_automl\data\1024_data\skl_1024.csv",
    ]
    data_2048 = [
        r"C:\Users\user\Desktop\madd_automl\data\2048_data\alz_2048.csv",
        r"C:\Users\user\Desktop\madd_automl\data\2048_data\cancer_clear_2048.csv",
        r"C:\Users\user\Desktop\madd_automl\data\2048_data\dislip_2048.csv",
        r"C:\Users\user\Desktop\madd_automl\data\2048_data\parkinson_2048.csv",
        r"C:\Users\user\Desktop\madd_automl\data\2048_data\resistance_2048.csv",
        r"C:\Users\user\Desktop\madd_automl\data\2048_data\skl_2048.csv",
    ]

    full_data = [data_1024, data_2048]
    for part_data in full_data:
        for data in part_data:
            name = Path(data).stem
            run_experiment(
                target='IC50', task='classification',
                data_path=data, save_name=name,
                time=60
            )
