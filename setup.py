import multiprocessing
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from distutils.command.clean import clean

from setuptools import Extension, find_packages, setup
from setuptools.command.build_py import build_py


# Env Variables
IS_DARWIN = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
RUN_BUILD_DEPS = any(arg in {"clean", "dist_info"} for arg in sys.argv)
COMPILE_FLAGS_FILE = os.path.join(BASE_DIR, "torch_mcpu", "_compile_flags.json")
CMAKE_STAGING_PACKAGE_DIR = os.path.join(BASE_DIR, "build", "cmake_install", "torch_mcpu")
IS_EDITABLE_BUILD = "editable_wheel" in sys.argv


def cmake_bool(value):
    normalized = str(value).strip().upper()
    if normalized in {"1", "ON", "TRUE", "YES", "Y"}:
        return "ON"
    if normalized in {"0", "OFF", "FALSE", "NO", "N"}:
        return "OFF"
    raise ValueError(f"Invalid CMake boolean value: {value!r}")


def cpp_bool(value):
    return "1" if cmake_bool(value) == "ON" else "0"


def get_mcpu_build_options():
    return {
        "TORCH_MCPU_ENABLE_MEMORY_PROTECTION": cmake_bool(
            os.getenv("TORCH_MCPU_ENABLE_MEMORY_PROTECTION", "OFF")
        ),
        "TORCH_MCPU_ENABLE_CPU_FALLBACK": cmake_bool(
            os.getenv("TORCH_MCPU_ENABLE_CPU_FALLBACK", "ON")
        ),
        "TORCH_MCPU_ENABLE_SYNC_KERNEL_LAUNCH": cmake_bool(
            os.getenv("TORCH_MCPU_ENABLE_SYNC_KERNEL_LAUNCH", "OFF")
        ),
    }


def write_compile_flags_file(build_options):
    definitions = {
        name: cpp_bool(value) for name, value in build_options.items()
    }
    payload = {
        "cmake_options": build_options,
        "definitions": definitions,
        "compile_flags": [
            f"-D{name}={value}" for name, value in definitions.items()
        ],
    }
    with open(COMPILE_FLAGS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def make_relative_rpath_args(path):
    if IS_DARWIN:
        return ["-Wl,-rpath,@loader_path/" + path]
    elif IS_WINDOWS:
        return []
    else:
        return ["-Wl,-rpath,$ORIGIN/" + path]


def get_pytorch_dir():
    # Disable autoload of the accelerator

    # We must do this for two reasons:
    # We only need to get the PyTorch installation directory, so whether the accelerator is loaded or not is irrelevant
    # If the accelerator has been previously built and not uninstalled, importing torch will cause a circular import error
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch

    return os.path.dirname(os.path.realpath(torch.__file__))


def build_deps():
    build_dir = os.path.join(BASE_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)
    if os.path.isdir(CMAKE_STAGING_PACKAGE_DIR):
        shutil.rmtree(CMAKE_STAGING_PACKAGE_DIR)
    build_options = get_mcpu_build_options()
    write_compile_flags_file(build_options)

    cmake_args = [
        "-DCMAKE_INSTALL_PREFIX="
        + os.path.realpath(CMAKE_STAGING_PACKAGE_DIR),
        "-DPYTHON_INCLUDE_DIR=" + sysconfig.get_paths().get("include"),
        "-DPYTORCH_INSTALL_DIR=" + get_pytorch_dir(),
        "-DTORCH_MCPU_ENABLE_MEMORY_PROTECTION="
        + build_options["TORCH_MCPU_ENABLE_MEMORY_PROTECTION"],
        "-DTORCH_MCPU_ENABLE_CPU_FALLBACK="
        + build_options["TORCH_MCPU_ENABLE_CPU_FALLBACK"],
        "-DTORCH_MCPU_ENABLE_SYNC_KERNEL_LAUNCH="
        + build_options["TORCH_MCPU_ENABLE_SYNC_KERNEL_LAUNCH"],
        "-DTORCH_MCPU_SHARED_USE_PREMAPPED="
        + cmake_bool(os.getenv("TORCH_MCPU_SHARED_USE_PREMAPPED", "OFF")),
    ]

    subprocess.check_call(
        ["cmake", BASE_DIR] + cmake_args, cwd=build_dir, env=os.environ
    )

    build_args = [
        "--build",
        ".",
        "--target",
        "install",
        "--config",  # For multi-config generators
        "Release",
        "--",
    ]

    if IS_WINDOWS:
        build_args += ["/m:" + str(multiprocessing.cpu_count())]
    else:
        build_args += ["-j", str(multiprocessing.cpu_count())]

    command = ["cmake"] + build_args
    subprocess.check_call(command, cwd=build_dir, env=os.environ)


class BuildPy(build_py):
    def run(self):
        # Setuptools may reuse build/lib.* between invocations.  That is cheap
        # for unchanged pure-Python files, but dangerous here: users often run
        # `python -m pip install .` after editing the mcpu Inductor backend, and
        # stale Python files in build/lib.* can otherwise be packaged into the
        # wheel.  Force only build_py's package-file copy; CMake outputs are
        # still managed by build_deps/copy_cmake_install_tree.
        self.force = True
        super().run()
        self.copy_cmake_install_tree()

    def copy_cmake_install_tree(self):
        build_package_dir = os.path.join(self.build_lib, "torch_mcpu")
        for name in ("include", "lib"):
            src = os.path.join(CMAKE_STAGING_PACKAGE_DIR, name)
            if os.path.isdir(src):
                shutil.copytree(
                    src,
                    os.path.join(build_package_dir, name),
                    dirs_exist_ok=True,
                )


class BuildClean(clean):
    def run(self):
        for i in ["build", "install", "torch_mcpu/lib"]:
            dirs = os.path.join(BASE_DIR, i)
            if os.path.exists(dirs) and os.path.isdir(dirs):
                shutil.rmtree(dirs)

        for dirpath, _, filenames in os.walk(os.path.join(BASE_DIR, "torch_mcpu")):
            for filename in filenames:
                if filename.endswith(".so"):
                    os.remove(os.path.join(dirpath, filename))


def main():
    write_compile_flags_file(get_mcpu_build_options())

    if not RUN_BUILD_DEPS:
        build_deps()

    if IS_WINDOWS:
        # /NODEFAULTLIB makes sure we only link to DLL runtime
        # and matches the flags set for protobuf and ONNX
        extra_link_args: list[str] = ["/NODEFAULTLIB:LIBCMT.LIB"] + [
            *make_relative_rpath_args("lib")
        ]
        # /MD links against DLL runtime
        # and matches the flags set for protobuf and ONNX
        # /EHsc is about standard C++ exception handling
        extra_compile_args: list[str] = ["/MD", "/FS", "/EHsc"]
    else:
        extra_link_args = [*make_relative_rpath_args("lib")]
        if IS_EDITABLE_BUILD:
            extra_link_args += [
                "-Wl,-rpath," + os.path.join(CMAKE_STAGING_PACKAGE_DIR, "lib")
            ]
        extra_compile_args = [
            "-Wall",
            "-Wextra",
            "-Wno-strict-overflow",
            "-Wno-unused-parameter",
            "-Wno-missing-field-initializers",
            "-Wno-unknown-pragmas",
            "-fno-strict-aliasing",
        ]

    ext_modules = [
        Extension(
            name="torch_mcpu._C",
            sources=["torch_mcpu/csrc/stub.c"],
            language="c",
            extra_compile_args=extra_compile_args,
            libraries=["torch_bindings"],
            library_dirs=[os.path.join(CMAKE_STAGING_PACKAGE_DIR, "lib")],
            extra_link_args=extra_link_args,
        )
    ]

    package_data = {
        "torch_mcpu": [
            "lib/*.so*",
            "lib/*.dylib*",
            "lib/*.dll",
            "lib/*.lib",
            "include/**/*.h",
            "include/**/*.hpp",
            "include/**/*.inl",
            "_compile_flags.json",
        ]
    }

    # LITERALINCLUDE START: SETUP
    setup(
        packages=find_packages(),
        package_data=package_data,
        ext_modules=ext_modules,
        cmdclass={
            "build_py": BuildPy,  # type: ignore[misc]
            "clean": BuildClean,  # type: ignore[misc]
        },
        include_package_data=False,
        entry_points={
            "torch.backends": [
                "torch_mcpu = torch_mcpu:_autoload",
            ],
        },
    )
    # LITERALINCLUDE END: SETUP


if __name__ == "__main__":
    main()
