from hatchling.builders.hooks.plugin.interface import BuildHookInterface
import subprocess
import sys
import importlib.util
import platform
import shutil

FEMWELL_GIT_URL = (
    "git+https://github.com/HelgeGehring/femwell.git"
    "@36e2ff1d8507e3839f29b5f14c298a091b463c49"
)


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if importlib.util.find_spec("femwell") is not None:
            print("femwell is already installed, skipping.")
            return

        print("Installing femwell from Git (no dependencies)...")
        print("Python executable:", sys.executable)

        if shutil.which("git") is None:
            _raise_git_missing()

        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--no-deps",
                FEMWELL_GIT_URL,
            ])
        except subprocess.CalledProcessError as exc:
            _raise_install_failed(exc)


def _raise_git_missing():
    msg = (
        "\n[imodulator] Cannot install femwell: 'git' was not found in PATH.\n"
    )
    if platform.system() == "Windows":
        msg += (
            "On Windows you can fix this in one of two ways:\n\n"
            "Option A – Install Git for Windows, then re-run pip install:\n"
            "  https://git-scm.com/download/win\n"
            "  (make sure to tick 'Add Git to PATH' during setup)\n\n"
            "Option B – Install femwell manually before installing imodulator:\n"
            f"  pip install --no-deps \"{FEMWELL_GIT_URL}\"\n"
            "  pip install imodulator\n"
        )
    else:
        msg += (
            "Please install git (e.g. 'sudo apt install git' or 'brew install git')\n"
            "and then re-run: pip install imodulator\n"
        )
    raise RuntimeError(msg)


def _raise_install_failed(exc):
    msg = (
        f"\n[imodulator] femwell installation failed: {exc}\n\n"
        "You can install femwell manually and then retry:\n"
        f"  pip install --no-deps \"{FEMWELL_GIT_URL}\"\n"
        "  pip install imodulator\n"
    )
    raise RuntimeError(msg) from exc
