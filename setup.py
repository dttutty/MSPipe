import os
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

gnnflow_lib = Extension(
    "libgnnflow", sources=[]
)

curdir = Path(__file__).resolve().parent


def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch must be installed before building gnnflow. "
            "Run `uv sync --no-install-project` first so the uv environment has torch."
        ) from exc
    return torch


def import_pybind11():
    try:
        import pybind11
    except ImportError as exc:
        raise RuntimeError(
            "pybind11 must be installed before building gnnflow. "
            "Run `uv sync --no-install-project` first so the uv environment has pybind11."
        ) from exc
    return pybind11


def get_cmake_bin():
    cmake_bin = "cmake"
    try:
        subprocess.check_output([cmake_bin, "--version"])
    except OSError:
        raise RuntimeError(
            "Cannot find CMake executable. "
            "Please install CMake and try again."
        )
    return cmake_bin


def split_paths(value):
    return [path for path in value.split(os.pathsep) if path]


def cmake_list(values):
    unique_values = []
    seen = set()
    for value in values:
        if not value:
            continue
        if value in seen:
            continue
        unique_values.append(value)
        seen.add(value)
    return ";".join(unique_values)


def get_python_library():
    libdir = sysconfig.get_config_var("LIBDIR")
    candidates = [
        sysconfig.get_config_var("LDLIBRARY"),
        sysconfig.get_config_var("LIBRARY"),
    ]
    for candidate in candidates:
        if not libdir or not candidate:
            continue
        library = os.path.join(libdir, candidate)
        if os.path.exists(library):
            return library
    return None


def candidate_native_prefixes():
    candidates = []
    for value in (
        os.environ.get("MSPIPE_NATIVE_PREFIX"),
        os.environ.get("CONDA_PREFIX"),
        str(Path.home() / "miniforge3" / "envs" / "mspipe-native"),
        str(Path.home() / "mambaforge" / "envs" / "mspipe-native"),
        str(Path.home() / ".conda" / "envs" / "mspipe-native"),
    ):
        if not value:
            continue
        if value in candidates:
            continue
        candidates.append(value)
    return [Path(value).expanduser() for value in candidates]


def find_native_prefix():
    required = (
        ("rmm", "lib/cmake/rmm/rmm-config.cmake"),
        ("absl", "lib/cmake/absl/abslConfig.cmake"),
        ("spdlog", "lib/cmake/spdlog/spdlogConfig.cmake"),
        ("fmt", "lib/cmake/fmt/fmt-config.cmake"),
    )
    for prefix in candidate_native_prefixes():
        if not prefix.exists():
            continue
        if all((prefix / relpath).exists() for _, relpath in required):
            return prefix
    return None


def append_cmake_arg(cmake_args, key, value):
    if value:
        cmake_args.append("-D{}={}".format(key, value))


class CustomBuildExt(build_ext):
    def build_extensions(self):
        torch = import_torch()
        pybind11 = import_pybind11()
        cmake_bin = get_cmake_bin()

        debug = os.environ.get("DEBUG", "0")
        config = 'Debug' if debug == "1" else 'Release'
        print("Building with CMake config: {}".format(config))

        ext_name = self.extensions[0].name
        build_dir = self.get_full_graph_with_reverse_edgespath(ext_name).replace(
            self.get_ext_filename(ext_name), '')
        build_dir = os.path.abspath(build_dir)

        os.makedirs(self.build_lib, exist_ok=True)
        os.makedirs(curdir / "build", exist_ok=True)

        prefix_paths = []
        if os.environ.get("CMAKE_PREFIX_PATH"):
            prefix_paths.extend(split_paths(os.environ["CMAKE_PREFIX_PATH"]))
        prefix_paths.append(pybind11.get_cmake_dir())
        prefix_paths.extend(split_paths(torch.utils.cmake_prefix_path))

        native_prefix = find_native_prefix()
        native_lib_dir = None
        if native_prefix is not None:
            native_prefix_str = str(native_prefix)
            prefix_paths.append(native_prefix_str)
            native_lib_dir = str(native_prefix / "lib")
            print("Using native dependency prefix: {}".format(native_prefix_str))

        cmake_args = [
            "-DCMAKE_BUILD_TYPE={}".format(config),
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            "-DPYTHON_EXECUTABLE:FILEPATH={}".format(sys.executable),
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(build_dir),
            "-DCMAKE_PREFIX_PATH={}".format(cmake_list(prefix_paths)),
            "-Dpybind11_DIR={}".format(pybind11.get_cmake_dir()),
        ]

        python_include_dir = sysconfig.get_path("include")
        if python_include_dir:
            cmake_args.extend([
                "-DPYTHON_INCLUDE_DIR={}".format(python_include_dir),
                "-DPYTHON_INCLUDE_DIRS={}".format(python_include_dir),
            ])

        python_library = get_python_library()
        if python_library:
            cmake_args.extend([
                "-DPYTHON_LIBRARY={}".format(python_library),
                "-DPYTHON_LIBRARIES={}".format(python_library),
            ])

        resolved_dependency_dirs = {
            "rmm_DIR": None,
            "absl_DIR": None,
            "spdlog_DIR": None,
            "fmt_DIR": None,
        }
        if native_prefix is not None:
            resolved_dependency_dirs.update({
                "rmm_DIR": str(native_prefix / "lib" / "cmake" / "rmm"),
                "absl_DIR": str(native_prefix / "lib" / "cmake" / "absl"),
                "spdlog_DIR": str(native_prefix / "lib" / "cmake" / "spdlog"),
                "fmt_DIR": str(native_prefix / "lib" / "cmake" / "fmt"),
            })

        for cmake_var, env_var in (
            ("rmm_DIR", "RMM_DIR"),
            ("absl_DIR", "ABSL_DIR"),
            ("spdlog_DIR", "SPDLOG_DIR"),
            ("fmt_DIR", "FMT_DIR"),
            ("CMAKE_CUDA_COMPILER", "CMAKE_CUDA_COMPILER"),
            ("CMAKE_C_COMPILER", "CMAKE_C_COMPILER"),
            ("CMAKE_CXX_COMPILER", "CMAKE_CXX_COMPILER"),
        ):
            value = os.environ.get(env_var)
            if not value:
                value = resolved_dependency_dirs.get(cmake_var)
            append_cmake_arg(cmake_args, cmake_var, value)

        if native_lib_dir:
            append_cmake_arg(cmake_args, "CMAKE_BUILD_RPATH", native_lib_dir)
            append_cmake_arg(cmake_args, "CMAKE_INSTALL_RPATH", native_lib_dir)

        cmake_build_args = [
            "--build",
            ".",
            "--config",
            config,
            "--parallel",
            str(os.cpu_count() or 1),
        ]

        try:
            subprocess.check_call(
                [cmake_bin, "..", *cmake_args],
                cwd=curdir / "build",
            )
            subprocess.check_call(
                [cmake_bin, *cmake_build_args],
                cwd=curdir / "build",
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "CMake build failed. If CMake cannot find rmm, absl, spdlog, or fmt, "
                "set MSPIPE_NATIVE_PREFIX, RMM_DIR, ABSL_DIR, SPDLOG_DIR, FMT_DIR, "
                "or extend CMAKE_PREFIX_PATH."
            ) from e

setup(
    ext_modules=[gnnflow_lib],
    cmdclass={"build_ext": CustomBuildExt},
)
