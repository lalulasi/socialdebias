"""Fail-fast import and resource checks for the external ENDEF environment."""

import argparse
import importlib
import subprocess
import sys
from pathlib import Path


MODULES = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "torch",
    "transformers",
    "tqdm",
    "nltk",
    "jieba",
    "spacy",
    "requests",
)


def check_imports():
    print("[1/5] Python package imports")
    for name in MODULES:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"  OK {name}={version}")


def check_versions():
    print("[2/5] NumPy/pandas compatibility")
    import numpy
    import pandas

    if numpy.__version__ != "1.26.4":
        raise RuntimeError(f"Expected numpy 1.26.4, got {numpy.__version__}")
    if pandas.__version__ != "1.5.3":
        raise RuntimeError(f"Expected pandas 1.5.3, got {pandas.__version__}")
    if not hasattr(pandas.DataFrame, "append"):
        raise RuntimeError("pandas.DataFrame.append is missing; ENDEF needs pandas 1.x")
    print("  OK pandas ABI and DataFrame.append")


def check_language_resources():
    print("[3/5] spaCy and NLTK resources")
    import nltk
    import spacy

    spacy.load("en_core_web_sm")
    nltk.data.find("tokenizers/punkt")
    # NLTK 3.9 uses punkt_tab; the second lookup makes failures explicit.
    nltk.data.find("tokenizers/punkt_tab")
    print("  OK en_core_web_sm, punkt, punkt_tab")


def check_endef_tree(root):
    print("[4/5] ENDEF source-tree imports")
    for subdir in ("ENDEF_en", "ENDEF_ch"):
        workdir = root / subdir
        if not (workdir / "grid_search.py").is_file():
            raise FileNotFoundError(f"Missing ENDEF source tree: {workdir}")
        code = (
            "import grid_search; "
            "from utils.dataloader import get_dataloader; "
            f"print('  OK {subdir} full imports')"
        )
        subprocess.run([sys.executable, "-c", code], cwd=workdir, check=True)


def check_torch(require_cuda=False):
    print("[5/5] PyTorch device")
    import torch

    print(f"  torch={torch.__version__}")
    print(f"  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")
    else:
        message = "PyTorch cannot see CUDA; ENDEF training requires a GPU environment"
        if require_cuda:
            raise RuntimeError(message)
        print(f"  WARNING: {message}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endef_root", type=Path, required=True)
    parser.add_argument("--require_cuda", action="store_true")
    args = parser.parse_args()
    root = args.endef_root.resolve()

    check_imports()
    check_versions()
    check_language_resources()
    check_endef_tree(root)
    check_torch(require_cuda=args.require_cuda)
    print("\nENDEF_ENV_READY=True")


if __name__ == "__main__":
    main()
