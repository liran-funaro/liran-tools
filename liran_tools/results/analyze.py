"""
Author: Liran Funaro <liran.funaro@gmail.com>
"""
import dataclasses
import functools
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from liran_tools.results import Experiment, ExperimentGroup


class AddTimeSeconds(Enum):
    SinceTimeSeries = "series"
    SinceExperimentBegin = "exp"
    Nothing = "no"


class Aggregator(Enum):
    Sum = "sum"
    Avg = "avg"


class Method(Enum):
    Rate = "rate"
    HistMean = "hist-mean",
    HistPercentile = "hist-percentile"
    Value = "value"


ByType = list[str] | tuple[str] | None


@dataclasses.dataclass
class DataConfig:
    method: Method = Method.Rate
    field: str = "throughput"
    value_field: str = "Throughput"
    percentile: float = 0.5
    aggregator: Aggregator = Aggregator.Avg
    timestamp_field: str = "timestamp"
    by: ByType = None
    window: str = "10s"
    filter: str = None
    add_time: AddTimeSeconds = AddTimeSeconds.SinceExperimentBegin


DEFAULT_CONFIG = DataConfig()
TimeField = "Time (seconds)"


def get_timestamp_col(df: pd.DataFrame, conf: DataConfig = DEFAULT_CONFIG):
    if conf.timestamp_field in df.columns:
        return df[conf.timestamp_field]
    else:
        return df.index.get_level_values(conf.timestamp_field)


def get_timeseries_min_time(*dfs: pd.DataFrame, conf: DataConfig = DEFAULT_CONFIG):
    assert len(dfs) > 0
    return min([get_timestamp_col(df, conf=conf).min() for df in dfs])


def add_time_seconds(e: Experiment, *dfs: pd.DataFrame, conf: DataConfig = DEFAULT_CONFIG):
    add_time = AddTimeSeconds(conf.add_time)
    if add_time.name == AddTimeSeconds.SinceTimeSeries.name:
        min_time = get_timeseries_min_time(*dfs, conf=conf)
    elif add_time.name == AddTimeSeconds.SinceExperimentBegin.name:
        min_time = e.min_time_no_tz
    else:
        return

    for df in dfs:
        df[TimeField] = (get_timestamp_col(df, conf=conf) - min_time).total_seconds()


def get_min_max_time(df: pd.DataFrame):
    assert len(df) > 0
    time_col = df[TimeField]
    return time_col.min(), time_col.max()


def _get_by_aggregator(by: list[str] | None = None):
    if by is None or len(by) == 0:
        return ""
    return f"by ({','.join(by)})"


def _field(conf: DataConfig, suffix: str | None = None, window=True):
    if conf.filter is None:
        field_filter = ""
    else:
        field_filter = f"{{{conf.filter}}}"
    name = conf.field
    if suffix is not None:
        name = f"{name}_{suffix}"
    if window is True:
        window = f"[{conf.window}]"
    else:
        window = ""
    return f"{name}{field_filter}{window}"


def get(e: Experiment, conf: DataConfig = DEFAULT_CONFIG) -> Optional[pd.DataFrame]:
    by_agg = _get_by_aggregator(conf.by)
    method = Method(conf.method)
    if method.name == Method.Rate.name:
        query = f"{conf.aggregator} {by_agg} (rate({_field(conf)}))"
    elif method.name == Method.HistMean.name:
        query = (
            f"("
            f"({conf.aggregator} {by_agg} (irate({_field(conf, 'sum')})))/"
            f"({conf.aggregator} {by_agg} (irate({_field(conf, 'count')})))"
            f")"
        )
    elif method.name == Method.HistPercentile.name:
        query = (
            f"{conf.aggregator} {_get_by_aggregator(conf.by)} ("
            f"histogram_quantile({conf.percentile}, rate({_field(conf, 'bucket')}))"
            f")"
        )
    elif method.name == Method.Value.name:
        query = f"{conf.aggregator} {_get_by_aggregator(conf.by)} ({_field(conf, window=False)})"
    else:
        raise ValueError(f"Unknown method: {conf.method}")
    df = e.query(query, value_field=conf.value_field)
    add_time_seconds(e, df, conf=conf)
    df.reset_index(inplace=True)
    return df


def le_to_hist(gdf: pd.DataFrame, conf: DataConfig = DEFAULT_CONFIG):
    gdf.sort_values("le", inplace=True)
    f = gdf[conf.value_field].diff()
    s = f.sum()
    if not np.isclose(s, 0, atol=1e-3):
        f /= s
    gdf[conf.value_field] = f
    return gdf


def get_hist(
        e: Experiment, conf: DataConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    by = ["le"]
    if conf.by:
        if "le" not in conf.by:
            by = ["le", *conf.by]
        else:
            by = conf.by

    df = e.query(
        f"{conf.aggregator} {_get_by_aggregator(by)} (rate({_field(conf, 'bucket')}))",
        value_field=conf.value_field,
    )

    df["le"] = pd.Categorical(df["le"], sorted(df["le"].unique(), key=lambda x: float(x)))
    df = df.groupby(df.index, group_keys=False).apply(lambda gdf: le_to_hist(gdf, conf))
    return df


def get_multi_hist(
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
            f"{conf.aggregator} {_get_by_aggregator(by)} (rate({k}_bucket[{conf.window}]))",
            value_field=conf.value_field,
        ) for k in fields
    }

    for k, df in dfs.items():
        df["le"] = pd.Categorical(df["le"], sorted(df["le"].unique(), key=lambda x: float(x)))
        dfs[k] = df.groupby(df.index, group_keys=False).apply(lambda gdf: le_to_hist(gdf, conf))

    return dfs


def get_all(e: Experiment, conf: list[DataConfig] = (DEFAULT_CONFIG,)) -> Optional[pd.DataFrame]:
    timestamp_field = conf[0].timestamp_field
    add_time = conf[0].add_time
    by = conf[0].by
    assert all(timestamp_field == c.timestamp_field for c in conf)
    assert all(add_time == c.add_time for c in conf)
    assert all(by == c.by for c in conf)

    dfs = [
        get(e, conf=dataclasses.replace(c, add_time=AddTimeSeconds.Nothing))
        for c in conf
    ]

    key = [timestamp_field]
    if by is not None:
        key.extend(by)

    for df in dfs:
        df.set_index(key, inplace=True)

    df = functools.reduce(lambda ldf, rdf: ldf.join(rdf, how="outer"), dfs)
    add_time_seconds(e, df, conf=conf[0])
    return df.reset_index()


def collect_all(eg: ExperimentGroup, conf: list[DataConfig] = (DEFAULT_CONFIG,)):
    return eg.collect(lambda e: get_all(e, conf=conf))
