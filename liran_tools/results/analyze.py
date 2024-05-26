"""
Author: Liran Funaro <liran.funaro@gmail.com>
"""
import dataclasses
from enum import Enum
from typing import Optional

import pandas as pd
from liran_tools.results import Experiment, ExperimentGroup


class AddTimeSeconds(Enum):
    SinceTimeSeries = "series"
    SinceExperimentBegin = "exp"
    Nothing = "no"


ByType = list[str] | tuple[str] | None


@dataclasses.dataclass
class DataConfig:
    throughput_field: str
    latency_field: str
    timestamp_field: str = "timestamp"
    by: ByType = None
    window: str = "10s"
    add_time: AddTimeSeconds = AddTimeSeconds.SinceExperimentBegin


DEFAULT_CONFIG = DataConfig("throughput_field", "latency_field")


def get_timestamp_col(df: pd.DataFrame, conf: DataConfig = DEFAULT_CONFIG):
    if conf.timestamp_field in df.columns:
        return df[conf.timestamp_field]
    else:
        return df.index.get_level_values(conf.timestamp_field)


def get_timeseries_min_time(*dfs: pd.DataFrame, conf: DataConfig = DEFAULT_CONFIG):
    assert len(dfs) > 0
    return min([get_timestamp_col(df, conf=conf).min() for df in dfs])


def add_time_seconds(e: Experiment, *dfs: pd.DataFrame, conf: DataConfig = DEFAULT_CONFIG):
    if conf.add_time == AddTimeSeconds.SinceTimeSeries:
        min_time = get_timeseries_min_time(*dfs, conf=conf)
    elif conf.add_time == AddTimeSeconds.SinceExperimentBegin:
        min_time = e.min_time_no_tz
    else:
        return

    for df in dfs:
        df["Time (seconds)"] = (get_timestamp_col(df, conf=conf) - min_time).total_seconds()


def _get_by_aggregator(by: list[str] | None = None):
    if by is None or len(by) == 0:
        return ""
    return f"by ({','.join(by)})"


def get_throughput(e: Experiment, conf: DataConfig = DEFAULT_CONFIG) -> Optional[pd.DataFrame]:
    df = e.query(
        f"avg {_get_by_aggregator(conf.by)} (rate({conf.throughput_field}[{conf.window}]) * 1e-3)",
        value_field="Throughput (K)"
    )
    add_time_seconds(e, df, conf=conf)
    df.reset_index(inplace=True)
    return df


def get_mean_latency(e: Experiment, conf: DataConfig = DEFAULT_CONFIG) -> Optional[pd.DataFrame]:
    by_agg = _get_by_aggregator(conf.by)
    df = e.query(
        f"(avg {by_agg} (irate({conf.latency_field}_sum[{conf.window}])))/"
        f"(avg {by_agg} (irate({conf.latency_field}_count[{conf.window}])))",
        value_field="Latency (seconds)",
    )
    add_time_seconds(e, df, conf=conf)
    df.reset_index(inplace=True)
    return df


def get_percentile(e: Experiment, percentile=0.99, conf: DataConfig = DEFAULT_CONFIG) -> Optional[pd.DataFrame]:
    df = e.query(
        f"avg {_get_by_aggregator(conf.by)} ("
        f"histogram_quantile({percentile}, rate({conf.latency_field}_bucket[{conf.window}]))"
        f")",
        value_field=f"{percentile * 100:.0f}-Latency (seconds)",
    )
    add_time_seconds(e, df, conf=conf)
    df.reset_index(inplace=True)
    return df


def get_rate(
        e: Experiment, field: str, conf: DataConfig = DEFAULT_CONFIG,
) -> Optional[pd.DataFrame]:
    df = e.query(f"sum {_get_by_aggregator(conf.by)} (rate({field}[{conf.window}]))")
    add_time_seconds(e, df, conf=conf)
    return df


def get_value(
        e: Experiment, field: str,conf: DataConfig = DEFAULT_CONFIG,
) -> Optional[pd.DataFrame]:
    df = e.query(f"sum {_get_by_aggregator(conf.by)} ({field})")
    add_time_seconds(e, df, conf=conf)
    return df


def le_to_hist(gdf: pd.DataFrame):
    gdf.sort_values("le", inplace=True)
    gdf["value"] = gdf["value"].diff()
    gdf["value"] /= gdf["value"].sum()
    return gdf


def get_hist(
        e: Experiment, fields: list[str], conf: DataConfig = DEFAULT_CONFIG,
) -> dict[str, pd.DataFrame]:
    by = ["le"]
    if conf.by:
        if "le" not in conf.by:
            by = ["le", *conf.by]
        else:
            by = conf.by

    dfs = {
        k: e.query(
            f"sum {_get_by_aggregator(by)} (rate({k}_bucket[{conf.window}]))"
        ) for k in fields
    }

    for k, df in dfs.items():
        df["le"] = pd.Categorical(df["le"], sorted(df["le"].unique(), key=lambda x: float(x)))
        dfs[k] = df.groupby(df.index, group_keys=False).apply(le_to_hist)

    return dfs


def get_all(e: Experiment, conf: DataConfig = DEFAULT_CONFIG) -> Optional[pd.DataFrame]:
    intermediate_conf = dataclasses.replace(conf, add_time=AddTimeSeconds.Nothing)
    dfs = [
        get_throughput(e, conf=intermediate_conf),
        get_mean_latency(e, conf=intermediate_conf),
        get_percentile(e, conf=intermediate_conf),
    ]

    for df in dfs:
        df.set_index([conf.timestamp_field], inplace=True)

    df = dfs[0].join(dfs[1]).join(dfs[2])
    add_time_seconds(e, df, conf=conf)
    return df.reset_index()


def collect_all(eg: ExperimentGroup, conf: DataConfig = DEFAULT_CONFIG):
    return eg.collect(lambda e: get_all(e, conf=conf))
