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

"""Script for plotting throughputs after benchmarks have been run.

NOTE: Since JAX and Torch cannot share the GPU, this script must be launched in
      a separate process. It loads the CSV results saved by `benchmark_main.py`
      and joins the two tables, before creating the throughput or speedup plots
      with seaborn.
"""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from benchmark_main import unpivot_runtimes

DIR = Path(__file__).parent / "outputs"
# DIR = Path("scripts/outputs/fused")

# If Path, read list[Path] from .txt file
PATHS: list[Path] | Path = [
    DIR / "TensorProductBenchmark.csv",
    DIR / "TorchTensorProductBenchmark.csv",
]

TITLE = "Tensor Product "
LOG_Y = True

PALETTE = dict(
    e3x="#38f",
    e3nn="#3aa",
    e3j="#a48",
    e3j_ops="#f49",
    torch_cuet="#3fa",
    torch_openeq="#2d9",
    e3nn_torch="#e53",
)

L_MAX_STYLE = "col"  # None  # "style"

RUNTIME = "runtime μs"


def main():
    """Meld benchmark results and draw joint plots."""
    df_long = join_runtimes(PATHS)
    filter_out = [
        "torch_cuet",
    ]
    plot_runtimes(df_long)
    plot_throughputs(df_long)
    plot_throughputs(df_long, filter_out)


def _get_paths() -> list[Path]:
    """Read .txt path list from $E3J_CSV_OUTPUTS or return default PATHS."""
    path_list = os.environ.get("E3J_CSV_OUTPUTS")
    if path_list is None:
        return PATHS
    with open(path_list, "r") as txt:
        return txt.readlines()


def _write_throughputs(df_long: pd.DataFrame) -> pd.DataFrame:
    sizeof_f32 = 4
    runtime_us = df_long[RUNTIME]
    througput_gb_s = (
        0.001
        * sizeof_f32
        * (
            df_long["batch_size"]
            * (df_long["dim_in"] + df_long["dim_out"])
            / runtime_us
        )
    )
    df_long["throughput GB/s"] = throughput
    return df_long


def join_runtimes(paths: list[Path] = PATHS):
    """Convert runtimes to long-form and join them."""
    dfs_long = []
    for csv_path in paths:
        df = pd.read_csv(csv_path, index_col=0)
        e3_keys = [key for key in df.keys() if key in PALETTE.keys()]
        df_long = unpivot_runtimes(df, e3_keys)
        dfs_long.append(df_long)
    df_long = pd.concat(dfs_long)

    df_long = _write_throughputs(df_long)

    df_long.to_csv(DIR / "Results_long.csv")
    return df_long


def plot_runtimes(
    df_long: pd.DataFrame,
    l_max: Literal["style"] | Literal["col"] | None = L_MAX_STYLE,
    **kws,
):
    if l_max is not None:
        kws[l_max] = "l_max"

    grid = sns.relplot(
        df_long,
        x="batch_size",
        y="runtime μs",
        hue="lib",
        kind="line",
        palette=PALETTE,
        **kws,
    )
    if LOG_Y:
        grid.set(xscale="log", yscale="log")
    plt.suptitle(TITLE + "runtimes\n\n ")

    sns.move_legend(grid, "right")

    # plt.tight_layout()
    plt.savefig(DIR / "TensorProduct.svg")


def plot_throughputs(
    df_long: pd.DataFrame,
    filter_out: list | None = None,
    l_max: Literal["style"] | Literal["col"] | None = L_MAX_STYLE,
    **kws,
):
    if l_max is not None:
        kws[l_max] = "l_max"

    suffix = "_filtered" if filter_out else ""

    filter_out = filter_out or []
    for lib in filter_out:
        df_long = df_long[df_long != lib]

    plt.suptitle(TITLE + "throughputs")

    grid = sns.relplot(
        df_long,
        x="batch_size",
        y="throughput GB/s",
        hue="lib",
        kind="line",
        palette=PALETTE,
        **kws,
    )
    if LOG_Y:
        grid.set(xscale="log", yscale="log")
    sns.move_legend(grid, "right")

    # plt.tight_layout()
    plt.savefig(DIR / f"TensorProduct_bw{suffix}.svg")


if __name__ == "__main__":
    main()
