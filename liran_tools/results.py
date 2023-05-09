"""
Author: Liran Funaro <liran.funaro@gmail.com>

Copyright (C) 2006-2023 Liran Funaro

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import datetime
import itertools
import os
import subprocess
import sys
import weakref
import re
from multiprocessing.pool import ThreadPool

import psutil
import yaml
from dateutil import parser
from pathlib import Path
from typing import List, Optional, Callable, Union, Tuple
import pandas as pd

from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame, PrometheusApiClientException
from tqdm.notebook import tqdm

SPACE_SPLITTER = re.compile(r"\s+", re.I | re.M)
NUM_SPLITTER = r = re.compile(r"(\d+)")
PROMETHEUS_PORT_RANGE = range(20_000, 30_000)
PROMETHEUS_PORT_ITER = itertools.cycle(PROMETHEUS_PORT_RANGE)
DATE_FORMAT = "%Y-%m-%d--%H:%M:%S"
MAIN_PATH = os.path.expanduser('~/workspace-data/results')
GROUP_EXP_PARAMETER_SHEET_FILE = "param-sheet.csv"
EXP_PARAMETERS_FILE = "log/exp.ini"


class NoSuchResultError(Exception):
    def __init__(self, name):
        super().__init__(f"No result with name: {name}")


def get_result_paths(res_dir: str, main_path: str = MAIN_PATH):
    res_path = Path(os.path.expanduser(main_path))
    res = filter(lambda s: s.is_dir(), (s.joinpath(res_dir) for s in res_path.iterdir()))
    return list(res)


def get_path_date(path):
    for p in Path(path).parts:
        # noinspection PyBroadException
        try:
            return datetime.datetime.strptime(p, DATE_FORMAT)
        except Exception:
            continue


def iter_date_path(*path: Path):
    for p in path:
        if not p.is_dir():
            continue
        d = get_path_date(p)
        if d is None:
            continue
        yield d, p


def _try_num(s: str) -> Union[int, str]:
    # noinspection PyBroadException
    try:
        return int(s)
    except Exception:
        return s


def path_sort_key(path_name: str) -> Tuple[Union[int, str]]:
    return tuple(map(_try_num, NUM_SPLITTER.split(path_name)))


def sort_path_by_num(*path: Path) -> List[Tuple[Tuple[Union[int, str]], Path]]:
    return sorted([(path_sort_key(p.name), p) for p in path])


def get_latest_path(*paths: Path, index=0, force_date=False):
    date_paths = list(iter_date_path(*paths))
    if len(date_paths) == 0:
        date_paths = [sub_p for p in paths for sub_p in iter_date_path(*p.iterdir())]

    if len(date_paths) > 0:
        _, date_paths = zip(*sorted(date_paths))
    else:
        if force_date:
            raise NoSuchResultError("no dates in paths")
        else:
            date_paths = paths

    index = len(date_paths) - 1 - index
    if index < 0:
        index = 0
    return date_paths[index]


def get_latest_result_path(res_dir: str, main_path: str = MAIN_PATH, index=0, force_date=False):
    paths = get_result_paths(res_dir, main_path)
    if len(paths) == 0:
        raise NoSuchResultError(res_dir)

    return get_latest_path(*paths, index=index, force_date=force_date)


def get_latest_sub_result_path(res_dir: Path, index=0, force_date=False):
    return get_latest_path(res_dir, index=index, force_date=force_date)


def get_log_lines_first_time(log_lines: List[str]):
    for line in log_lines:
        try:
            return parser.parse(SPACE_SPLITTER.split(line)[0])
        except parser.ParserError:
            continue


def get_log_min_max_time(log_file: Path):
    with log_file.open('r') as f:
        lines = f.readlines()
        return get_log_lines_first_time(lines), get_log_lines_first_time(lines[::-1])


def find_all_servers(port_range: Union[range, set, int] = PROMETHEUS_PORT_RANGE):
    if isinstance(port_range, int):
        port_range = {port_range}

    for child in psutil.Process().children(recursive=True):
        try:
            ports = {c.laddr.port for c in child.connections()}.intersection(port_range)
            if ports:
                yield child, ports
        except (psutil.ZombieProcess, psutil.NoSuchProcess):
            pass


def kill_all_servers(port_range: Union[range, set, int] = PROMETHEUS_PORT_RANGE):
    for p, _ in find_all_servers(port_range):
        try:
            p.terminate()
        except psutil.ZombieProcess:
            pass


def display_servers(port_range: Union[range, set, int] = PROMETHEUS_PORT_RANGE):
    servers = list(find_all_servers(port_range))
    return pd.DataFrame([
        [s.name(), s.pid, s.status(), str(Path(s.cwd()).relative_to(MAIN_PATH)),
         datetime.datetime.fromtimestamp(s.create_time()), list(p)]
        for s, p in servers
    ], columns=['name', 'pid', 'status', 'cwd', 'started', 'ports'])


def read_prop_file(yaml_path: Path) -> dict:
    with yaml_path.open('rb') as f:
        return yaml.safe_load(f)


def get_exp_csv(path: Path,
                group_sheet_filename=GROUP_EXP_PARAMETER_SHEET_FILE,
                exp_parameters_filename=EXP_PARAMETERS_FILE):
    csv_path = os.path.join(path, group_sheet_filename)
    if os.path.isfile(csv_path):
        return pd.read_csv(csv_path)

    exp_group_name = path.name

    exp_path = list(path.iterdir())
    exp_prop_path = [get_latest_sub_result_path(p).joinpath(exp_parameters_filename) for p in exp_path]
    if all(p.is_file() for p in exp_prop_path):
        exp_prop = [read_prop_file(p) for p in exp_prop_path]
        for cur_path, prop in zip(exp_path, exp_prop):
            prop['exp_group_name'] = exp_group_name
            prop['name'] = cur_path.name
        return pd.DataFrame(exp_prop)

    df = [[exp_group_name, p.name, *t] for t, p in sort_path_by_num(*exp_path)]
    max_len = max(len(row) for row in df)
    return pd.DataFrame(df, columns=['exp_group_name', 'name', *(f'f{i}' for i in range(max_len - 2))])


class ExperimentGroup:
    def __init__(self, name, result_index=0,
                 group_sheet_filename=GROUP_EXP_PARAMETER_SHEET_FILE,
                 exp_parameters_filename=EXP_PARAMETERS_FILE):
        self.name = name
        self.path = get_latest_result_path(name, index=result_index)
        self.exp = get_exp_csv(self.path, group_sheet_filename, exp_parameters_filename)
        self.exp_cache = {}

    def __repr__(self):
        return f'{self.__class__.__name__}({self.path.relative_to(MAIN_PATH)})'

    @classmethod
    def _next_port(cls):
        return next(PROMETHEUS_PORT_ITER)

    def __getitem__(self, item):
        if isinstance(item, int):
            idx = self.exp.iloc[item].name
        elif isinstance(item, str):
            idx = self.exp[self.exp['name'] == item].index[0]
        else:
            raise KeyError("Key must be integer or string.")

        name = self.exp.loc[idx]['name']
        cache_key = (str(self.path), name)
        exp = self.exp_cache.get(cache_key, None)
        if exp is None:
            p = get_latest_sub_result_path(self.path.joinpath(name))
            exp = Experiment(p, port=self._next_port())
            self.exp_cache[cache_key] = exp
        return exp

    def __len__(self):
        return len(self.exp)

    def iterexp(self):
        for _, row in self.exp.iterrows():
            try:
                e = self[row['name']]
                if e.is_executed:
                    yield row, e
            except NoSuchResultError as e:
                print(e, file=sys.stderr)

    def collect(self, exp_callback: Callable[['Experiment'], pd.DataFrame]):
        dfs = []
        with ThreadPool(32) as e:
            m = e.imap(lambda row_exp: (row_exp[0], exp_callback(row_exp[1])), self.iterexp())
            for row, df in tqdm(m, total=len(self)):
                if df is None:
                    continue
                for k, v in row.items():
                    df[k] = v
                dfs.append(df)
        if not dfs:
            return None
        return pd.concat(dfs)

    def kill_servers(self):
        kill_all_servers({exp.port for exp in self.exp_cache.values()})


class Experiment:
    def __init__(self, name, port=PROMETHEUS_PORT_RANGE[-1], result_index=0):
        self.name = name
        self.path = get_latest_result_path(name, index=result_index)
        self.log = self.path.joinpath("log")
        self.metrics = self.path.joinpath("metrics")
        self.port = port

        self._logs_min_max_time = None
        self._min_time = None
        self._workload_start_time = None
        self._max_time = None
        self._workload_end_time = None
        self._server: Optional[subprocess.Popen] = None
        self._prom: Optional[PrometheusConnect] = None

        weakref.finalize(self, self.kill_this_server)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.path.relative_to(MAIN_PATH)})'

    @property
    def is_executed(self):
        if not (self.log.is_dir() and len(list(self.log.iterdir())) > 0) and (
                self.metrics.is_dir() and len(list(self.metrics.iterdir())) > 0
        ):
            return False

        return self.min_time < self.max_time

    def logs(self):
        logs = self.path.joinpath("log")
        return [log_file for log_file in logs.iterdir() if log_file.match("*.log")]

    def benchmark_logs(self):
        return [log_file for log_file in self.logs() if 'prometheus' not in log_file.name]

    @property
    def logs_min_max_time(self):
        if self._logs_min_max_time is not None:
            return self._logs_min_max_time

        try:
            self._logs_min_max_time = {
                log_file.name: get_log_min_max_time(log_file)
                for log_file in self.benchmark_logs()
            }
        except Exception as e:
            print(self, "Failed reading time from logs:", e, file=sys.stderr)
            self._logs_min_max_time = {}

        return self._logs_min_max_time

    def _calc_min_max(self):
        try:
            min_time, max_time = zip(*self.logs_min_max_time.values())
            self._min_time = min(filter(None, min_time))
            self._max_time = max(filter(None, max_time))
        except Exception as e:
            print(self, "Failed reading time from logs:", e, file=sys.stderr)
            self._min_time = get_path_date(self.path)
            self._max_time = datetime.datetime.now()

    @property
    def min_time(self) -> datetime.datetime:
        if self._min_time is None:
            self._calc_min_max()
        return self._min_time

    @property
    def max_time(self) -> datetime.datetime:
        if self._max_time is None:
            self._calc_min_max()
        return self._max_time

    def start_server(self, save_output=False):
        if self.is_server_alive():
            return

        self.kill_server()

        stdout = subprocess.PIPE if save_output else subprocess.DEVNULL
        stderr = subprocess.PIPE if save_output else subprocess.DEVNULL

        self._server = subprocess.Popen([
            "prometheus",
            "--storage.tsdb.path=metrics",
            f"--config.file={os.path.abspath('prometheus-ro.yml')}",
            f"--web.listen-address=localhost:{self.port}"
        ], cwd=self.path, stdout=stdout, stderr=stderr)

    def is_server_alive(self):
        return self._server is not None and self._server.poll() is None

    def kill_this_server(self):
        if self.is_server_alive():
            self._server.kill()

    def kill_server(self):
        self.kill_this_server()
        kill_all_servers(self.port)

    def dump_server_stderr(self):
        if self._server is None:
            return
        while True:
            b = self._server.stderr.readline()
            if not b:
                break
            print(str(b, 'utf8'), end="")

    @property
    def prom_api(self):
        if self._prom is not None:
            return self._prom
        self._prom = PrometheusConnect(url=f"http://localhost:{self.port}/")
        return self._prom

    def _query(self, query, start_time, end_time):
        for step in ['10s', '30s', '1m']:
            try:
                return self.prom_api.custom_query_range(
                    query,
                    start_time=start_time,
                    end_time=end_time,
                    step=step,
                )
            except PrometheusApiClientException as e:
                exception = e

        raise exception

    def query(self, query):
        start_time = self.min_time - datetime.timedelta(minutes=1)
        end_time = self.max_time + datetime.timedelta(minutes=10)

        self.start_server()
        m = self._query(
            query,
            start_time=start_time,
            end_time=end_time,
        )
        if m:
            m = MetricRangeDataFrame(m)
            return m.sort_index()
        else:
            return None


def to_utc(t: datetime.datetime):
    return t.astimezone(datetime.timezone.utc).replace(tzinfo=None)
