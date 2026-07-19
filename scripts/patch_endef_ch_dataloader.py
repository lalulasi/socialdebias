#!/usr/bin/env python3
"""Make ENDEF_ch's dataloader accept equivalent numeric and string labels."""

import argparse
import os
import shutil
from pathlib import Path


ORIGINAL_FRAGMENT = (
    "    label = torch.tensor(df_data['label'].apply(lambda c: label_dict[c])"
    ".astype(int).to_numpy())"
)

PATCHED_FRAGMENT = """    def normalize_endef_label(value):
        # ENDEF's original Chinese datasets use real/fake, while converted
        # datasets commonly use the equivalent 0/1 representation.
        if value in label_dict:
            return label_dict[value]
        try:
            normalized = int(value)
        except (TypeError, ValueError) as exc:
            raise KeyError(f'unsupported label: {value!r}') from exc
        if normalized not in (0, 1):
            raise KeyError(f'unsupported label: {value!r}')
        return normalized

    label = torch.tensor(df_data['label'].apply(normalize_endef_label).astype(int).to_numpy())"""


def patch_dataloader(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ENDEF_ch dataloader not found: {path}")

    source = path.read_text(encoding="utf-8")
    if "def normalize_endef_label(value):" in source:
        compile(source, str(path), "exec")
        print(f"ALREADY_PATCHED {path}")
        print("ENDEF_CH_DATALOADER_READY=True")
        return False

    occurrences = source.count(ORIGINAL_FRAGMENT)
    if occurrences != 1:
        raise RuntimeError(
            f"expected exactly one original label conversion in {path}, found {occurrences}; "
            "refusing to modify an unknown ENDEF version"
        )

    patched = source.replace(ORIGINAL_FRAGMENT, PATCHED_FRAGMENT)
    compile(patched, str(path), "exec")

    backup = path.with_name(path.name + ".socialdebias.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"BACKUP {backup}")

    temporary = path.with_name(path.name + ".socialdebias.tmp")
    temporary.write_text(patched, encoding="utf-8")
    shutil.copymode(path, temporary)
    temporary.replace(path)
    print(f"PATCHED {path}")
    print("ENDEF_CH_DATALOADER_READY=True")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endef_ch_root",
        type=Path,
        default=os.environ.get("ENDEF_CH_ROOT"),
        help="ENDEF_ch root; defaults to ENDEF_CH_ROOT",
    )
    args = parser.parse_args()
    if args.endef_ch_root is None:
        parser.error("set ENDEF_CH_ROOT or pass --endef_ch_root")
    patch_dataloader(args.endef_ch_root / "utils" / "dataloader.py")


if __name__ == "__main__":
    main()
