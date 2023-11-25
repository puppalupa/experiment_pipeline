import pandas as pd
import config as cfg
from metric_builder import _load_yaml_preset
from report import build_experiment_report
import time

logger = cfg.logger
start_time = time.time()

df = pd.read_parquet(f'data/parquet/df.parquet')

logger.info("Data loaded")

experiment_report = build_experiment_report(
    df=df,
    metric_config=_load_yaml_preset()
)
experiment_report.to_csv(f"experiment_report.csv")

cfg.logger.info(time.time() - start_time)
