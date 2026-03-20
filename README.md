# MSPipe

This repository is the official implementation of MSPipe: Efficient Temporal GNN Training via Staleness-aware Pipeline

## Install

Our development environment:

- Ubuntu 20.04+
- Python 3.10
- CUDA 11.8
- cmake 3.18+

Dependencies:

- PyTorch 2.2.2 + cu118
- DGL 2.4.0 + cu118
- native CMake packages for `spdlog`, `absl`, `fmt`, and `rmm`

`uv` is the supported way to manage the Python environment for this repository.

On a fresh machine, bootstrap the non-Python build dependencies with:

```sh
bash scripts/setup_native_deps.sh install
```

By default the script installs Miniforge under `~/miniforge3` and creates
`~/miniforge3/envs/mspipe-native`. Set `MINIFORGE_PREFIX`, `NATIVE_ENV_NAME`,
or `CUDA_VERSION` before running it if you want different locations or a
different CUDA label.
If you already created `mspipe-native` with an older script version, rerun the
`install` command to refresh the `rmm` / `thrust` / `cub` toolchain.
`setup.py` will automatically detect the active conda environment or the default
`~/miniforge3/envs/mspipe-native` prefix during `uv sync`, so you do not need
to manually activate `mspipe-native` for the common case.

If you already have those native dependencies installed somewhere else, point
CMake at them explicitly:

```sh
export MSPIPE_NATIVE_PREFIX=/path/to/native/prefix
```

You can still use `RMM_DIR`, `ABSL_DIR`, `SPDLOG_DIR`, `FMT_DIR`, or
`CMAKE_PREFIX_PATH` if you need finer-grained overrides.

Create and sync the uv environment:

```sh
git submodule update --init --recursive
uv python install 3.10
uv sync --no-install-project
```

The command above creates a uv-managed `.venv` and installs the Python dependency stack.

If you already have a matching `libgnnflow*.so` in the repository root, you can stop here and run
the project with:

```sh
export PYTHONPATH=$PWD
```

Building `gnnflow` from source is optional and requires a compatible C++/CUDA dependency set.
The dynamic graph memory resource options are `cuda`, `unified`, and `pinned`.
Use `pinned` if you need host-visible graph storage for block offload.

If your machine has that compatible toolchain, build and install `gnnflow` editable with:

```sh
uv sync
```

For debug mode:

```sh
DEBUG=1 uv sync
```

Compile the TGL presample extension:

```sh
uv run python tgl/setup_tgl.py build_ext --inplace
```

## Prepare data

```sh
uv run bash -lc 'cd scripts && ./download_data.sh'
```

## Train

**MSPipe**

Training [TGN](https://arxiv.org/pdf/2006.10637v2.pdf) model on the REDDIT dataset with MSPipe on 4 GPUs.

```sh
uv run bash -lc 'cd scripts && ./run_offline.sh TGN REDDIT 4'
```

**Presample (TGL)** 

Training [TGN](https://arxiv.org/pdf/2006.10637v2.pdf) model on the REDDIT dataset with Presample on 4 GPUs.

```sh
uv run bash -lc 'cd tgl && ./run_tgl.sh TGN REDDIT 4'
```



**Distributed training**

Training TGN model on the GDELT dataset on more than 1 servers, each server is required to do the following step:

1. change the `INTERFACE` to your netcard name (can be found using`ifconfig`)
2. change the
   - `HOST_NODE_ADDR`: IP address of the host machine
   - `HOST_NODE_PORT`: The port of the host machine
   - `NNODES`: Total number of servers
   - `NPROC_PER_NODE`: The number of GPU for each servers

```sh
uv run bash -lc 'cd scripts && ./run_offline_dist.sh TGN GDELT'
```
