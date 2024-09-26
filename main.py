import subprocess
import sys
import argparse
from pathlib import Path


def main(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--in-fpath", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--blur-kernel-fpath", type=str)
    parser.add_argument("--slice-thickness", type=str)
    parser.add_argument("--patch-sampling", type=str, default="gradient")
    parser.add_argument("--suffix", type=str, default="_smore4")
    parser.add_argument("--gpu-id", type=str, default="0")

    args = parser.parse_args(args if args is not None else sys.argv[1:])

    in_fpath = Path(args.in_fpath).resolve()
    subj_id, ext = in_fpath.name.split(".", maxsplit=1)

    out_dir = Path(args.out_dir).resolve() / subj_id
    out_fpath = out_dir / f"{subj_id}{args.suffix}.{ext}"
    weight_dir = out_dir / "weights"

    train_cmd = [
        "smore-train",
        "--in-fpath",
        in_fpath,
        "--weight-dir",
        weight_dir,
        "--gpu-id",
        args.gpu_id,
        "--patch-sampling",
        args.patch_sampling,
        "--verbose",
    ]

    # Handle optional arguments
    if args.slice_thickness is not None:
        train_cmd.extend(["--slice-thickness", args.slice_thickness])
    if args.blur_kernel_fpath is not None:
        train_cmd.extend(["--blur-kernel-fpath", args.blur_kernel_fpath])

    test_cmd = [
        "smore-test",
        "--in-fpath",
        in_fpath,
        "--out-fpath",
        out_fpath,
        "--weight-dir",
        weight_dir,
        "--gpu-id",
        args.gpu_id,
        "--verbose",
    ]

    subprocess.run(train_cmd)
    subprocess.run(test_cmd)
