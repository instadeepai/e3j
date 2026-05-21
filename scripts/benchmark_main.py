# Copyright (c) 2026 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script for running multiple benchmarks, each on a grid of hyperparameters.

The grid of parameters is specified as in the example `scripts/config.yaml` file.
Runtime results are stored as CSV for reuse, while first plots can be saved as SVG.
See the `plot_speedup.py` script for more metrics.

TODO: support basic CLI arguments, such as CONFIG_PATH.
"""

import gc
import itertools
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import jax
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from benchmarks.benchmark_harmonics import HarmonicsBenchmark, HarmonicsGradBenchmark
from benchmarks.benchmark_linear import LinearBenchmark
from benchmarks.benchmark_symmetric_contraction import SymmetricContractionBenchmark
from benchmarks.benchmark_tensor_product import (
    TensorProductBenchmark,
    TensorProductGradBenchmark,
    TorchTensorProductBenchmark,
)
from benchmarks.e3_benchmark import E3Benchmark
from benchmarks.utils import dict_grid, dict_zip  # noqa

from e3j.utils import config, set_ram_limits

# Make sure Python is aware of RAM limits in container
# NOTE(armand): Needed to comment this for cuequivariance to work
# set_ram_limits()

# --- Main parameters ---

CONFIG_PATH = Path(__file__).parent / "config.yaml"  # "test_config.yaml"
DEBUG = 0

OUTPUT_DIR = Path(__file__).parent / "outputs"  # "TEST"

# Hack: prefix of csv files is f'{key.split(" ")[0].capitalize()'
#       => "Throughput-TensorProductBenchmark.{csv,svg}"
#       We still need to units in legends though, as returned by utils.timed_loop.
METRICS_MAP = [("Runtime", "runtime µs"), ("Throughput", "throughput GB/s")]

# Saves list of output csvs, for downstream merging of Torch+JAX tables.
if not os.environ.get("E3J_CSV_OUTPUTS"):
    os.environ["E3J_CSV_OUTPUTS"] = str(OUTPUT_DIR / "CSV_OUTPUTS.txt")

# ---

config(debug_level=DEBUG)

PANDAS_OPTIONS = {
    "display.precision": 1,
}
for k, opt_k in PANDAS_OPTIONS.items():
    pd.set_option(k, opt_k)

# E3Benchmark modules (accessible by names later on)
AVAILABLE_BENCHMARKS: list[E3Benchmark] = [
    HarmonicsBenchmark,
    HarmonicsGradBenchmark,
    LinearBenchmark,
    TensorProductBenchmark,
    TensorProductGradBenchmark,
    TorchTensorProductBenchmark,
    SymmetricContractionBenchmark,
]

# Assign E3Benchmark-specific maps from grid parameters to factory arguments
BENCHMARK_PARSERS: dict[str, Callable[[dict], dict]] = {}


def main():
    """Loop over benchmarks and save CSV + SVG results.

    Additionally, save names of CSV files as a `.txt` list for downstream
    scripts. This is e.g. necessary to aggregate JAX + Torch benchmarks
    which have to run in different processes.
    """
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir()

    with open(CONFIG_PATH, "r") as stream:
        cfg = yaml.safe_load(stream)

    csv_outputs = benchmark_all(cfg)
    plot_all(cfg)
    save_git_revision()

    with open(os.environ["E3J_CSV_OUTPUTS"], "w") as txt:
        txt.writelines(str(csv) + "\n" for csv in csv_outputs)


def get_benchmark_dir(bmk_name: str) -> Path:
    name = bmk_name.replace("Benchmark", "")
    bmk_dir = OUTPUT_DIR / name
    if not bmk_dir.exists():
        bmk_dir.mkdir()
    return bmk_dir


def _generate_plot_title(benchmark_name: str):
    # Remove "Benchmark"
    benchmark_name = benchmark_name[: benchmark_name.find("Benchmark")]
    # Turn camel case into title
    return "".join(
        " " + x if (x.isupper()) & (k > 0) else x for k, x in enumerate(benchmark_name)
    )


def benchmark_module(
    module: E3Benchmark,
    params: Iterable[dict],
    batch_sizes: Iterable[int],
    parse_fn: Callable[[dict], dict] | None = None,
    metadata: bool = False,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Benchmark a single module for a collection of hyper-parameters.

    The original `params` rows will be returned with new columns
    containing the average runtimes in microseconds.

    The `parse_fn` feeds keyword arguments to the `E3Benchmark`
    from any `param` row. This is especially useful when used in
    conjunction with `dict_grid`.
    """
    metrics = defaultdict(lambda: defaultdict(lambda: []))
    if parse_fn is None:
        parse_fn = lambda p: p

    # TODO: logic can be simplified a lot as DataFrame constructor
    #       accepts `rows: list[dict]`
    for param in params:

        if config().debug_level >= 2:
            # launch parameters will be logged at every kernel call,
            # disable batch_size iteration to relieve stdout.
            batch_sizes = batch_sizes[:1]
            param["n_it"] = 1

        row0 = dict(**param)
        benchmark = module(
            batch_size=128,  # placeholder
            **parse_fn(param),
        )
        if verbose:
            print(f"\n> params {param}: ", end="")

        for batch_size in batch_sizes:
            # Note: we update batch_size to avoid avoid recomputing coefs,
            #       cleaner would be to make it an argument of .run()
            benchmark.batch_size = batch_size
            metrics_row = benchmark.run(write=DEBUG >= 1)
            for key, metric in metrics_row.items():
                # Write results and parameters to table
                row = dict(
                    batch_size=batch_size,
                    # FIXME: error prone, remove
                    # dim_in=benchmark.dim_in,
                    # dim_out=benchmark.dim_out,
                    **metric,
                    **row0,
                )
                for k, vk in row.items():
                    metrics[key][k].append(vk)
                print(".", end="")
        print("\n")

        jax.clear_caches()
        gc.collect()

    return {key: pd.DataFrame(metric) for key, metric in metrics.items()}


def benchmark_all(cfg: dict) -> list[Path]:
    """Run the available module benchmarks over collections of hyper-parameters

    The `cfg` object can be specified as a `.yaml` file, see `benchmark.yaml`
    for the expected configuration structure.
    """
    # available : dict[str, E3Benchmark]
    available = {bmk.__name__: bmk for bmk in AVAILABLE_BENCHMARKS}
    # accumulate csv paths for downstream scripts
    csv_outputs = []

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir()

    for bmk_name, bmk_cfg in cfg["modules"].items():

        if bmk_cfg.get("skip"):
            continue

        # Map key to E3Benchmark
        e3_bmk = available[bmk_name]
        name = bmk_name.replace("Benchmark", "")
        print("\n", name, "=" * len(name), sep="\n")

        # Generate hyperparameter grid
        batch_sizes = bmk_cfg["grid"].pop("batch_size")
        params = dict_grid(bmk_cfg["grid"])

        # Loop over hyperparameters and save runtimes to csv
        df_dict = benchmark_module(
            e3_bmk,
            params,
            batch_sizes,
            parse_fn=BENCHMARK_PARSERS.get(bmk_name),
        )
        for key, df in df_dict.items():
            df_summary = df[df["batch_size"] == max(batch_sizes)]
            header = f"{key} summary"
            print(header, "-" * len(header), sep="\n")
            print(df_summary)
            prefix = key.split(" ")[0].capitalize()
            bmk_dir = get_benchmark_dir(bmk_name)
            csv_path = bmk_dir / f"{prefix}-{name}.csv"
            print(f"-> saving full table to {bmk_dir.name}/{csv_path.name}\n")
            df.to_csv(csv_path)
            csv_outputs.append(csv_path)

    return csv_outputs


def plot_all(cfg: dict):
    """Plot benchmark results with seaborn.

    The `cfg` object is shared with `benchmark_all`. We could potentially support an S3
    output path in the future, to automatically fetch results from previous experiments
    and plot results locally.
    """
    # available : dict[str, E3Benchmark]
    available = {bmk.__name__: bmk for bmk in AVAILABLE_BENCHMARKS}

    for (key, value), (bmk_name, bmk_cfg) in itertools.product(
        METRICS_MAP, cfg["modules"].items()
    ):

        name = bmk_name.replace("Benchmark", "")
        csv_path = get_benchmark_dir(bmk_name) / f"{key}-{name}.csv"

        if not csv_path.exists():
            if not bmk_cfg.get("skip"):
                print(f"WARNING: could not find {csv_path}")
            continue

        # Map key to E3Benchmark
        e3_bmk = available[bmk_name]

        # Load and unpivot dataframe (to long format)
        df = pd.read_csv(csv_path, index_col=0)
        df_long = unpivot_table(df, e3_bmk.keys, value=value)
        print(df_long)

        # Plot runtimes dataframe using seaborn
        sns_kwargs = bmk_cfg["seaborn"].copy() if "seaborn" in bmk_cfg else {}

        col = sns_kwargs.pop("col") if "col" in sns_kwargs else "l_max"
        grid = sns.relplot(
            df_long,
            x="batch_size",
            y=value,
            col=col,
            hue="lib",
            kind="line",
            palette=cfg["palette"],
            errorbar=None,
            **sns_kwargs,
        )
        if bmk_cfg.get("log_xy"):
            grid.set(xscale="log", yscale="log")
        title = _generate_plot_title(bmk_name)
        plt.suptitle(title)
        # Save plots to requested formats
        for fmt in cfg["formats"]:
            bmk_dir = get_benchmark_dir(bmk_name)
            name = bmk_name.replace("Benchmark", "")
            fig_path = bmk_dir / f"{key}-{name}.{fmt}"
            plt.tight_layout()
            plt.savefig(fig_path)


def unpivot_table(
    df: pd.DataFrame,
    keys: Iterable[str] = ("e3nn", "e3x", "e3j"),
    value: str = "runtime µs",
    id_vars: Iterable[str] | None = None,
):
    """Transform runtimes DataFrame to long-form."""
    if id_vars is None:
        keep_col = lambda k: (k not in keys and k[:7] != "Unnamed")  # noqa: E731
        id_vars = [k for k in df.keys() if keep_col(k)]

    return pd.melt(
        df,
        id_vars=id_vars,
        value_vars=keys,
        value_name=value,
        var_name="lib",
    )


def save_git_revision():
    sha = os.environ.get("VCS_SHA")
    if not sha:
        sha = os.popen("git rev-parse HEAD").read().strip()
    with open(OUTPUT_DIR / f"GIT_REVISION_{sha}.txt", "w") as f:
        f.write(f"https://github.com/instadeepai/e3j/commit/{sha}")


# --- Ad hoc maps from parameter grid to benchmark arguments ---


def linear_parse_fn(param: dict) -> dict:
    """Return equal source and target from `(mul, l_max)` fields."""
    mul = param.pop("mul")
    l_max = param.pop("l_max")
    param.update(
        source=(mul, l_max),
        target=(mul, l_max),
    )
    return param


def otimes_parse_fn(param: dict) -> dict:
    """Return equal source pairs from `(l_max)` field."""
    l_max = param.pop("l_max")
    param.update(
        source=((1, l_max), (1, l_max)),
        target=(1, l_max),
    )
    return param


def sym_cont_parse_fn(param: dict) -> dict:
    """Return equal source and target from `(l_max)` fields."""
    l_max = param.pop("l_max")
    l_max_out = param.pop("l_max_out")
    param.update(
        source=((1, l_max)),
        target=(1, l_max_out),
    )
    return param


BENCHMARK_PARSERS["LinearBenchmark"] = linear_parse_fn
BENCHMARK_PARSERS["TensorProductBenchmark"] = otimes_parse_fn
BENCHMARK_PARSERS["TensorProductGradBenchmark"] = otimes_parse_fn
BENCHMARK_PARSERS["TorchTensorProductBenchmark"] = otimes_parse_fn
BENCHMARK_PARSERS["SymmetricContractionBenchmark"] = sym_cont_parse_fn
BENCHMARK_PARSERS["HarmonicsGradBenchmark"] = lambda params: (
    {"primitive": HarmonicsBenchmark(**params)}
)


if __name__ == "__main__":
    main()
