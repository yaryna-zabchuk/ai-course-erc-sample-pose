"""K-fold cross-validation over recording sessions.

The test split has only ~20 human-labelled probes, so single-split numbers
move by several points from one run to the next. Here every session is the
test split once and the metrics are averaged over the folds.

    python -m erc_pose.crossval --export data/raw/semi_supervised_fixed --configs pose obb --fuse --name base
    python -m erc_pose.crossval --export data/raw/semi_supervised_fixed --configs pose obb --fuse --name semi \
        --extra-train data/pseudo
    python -m erc_pose.crossval --name mixed --mix semi base   # semi OBB directed by base pose
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from erc_pose.data.build_dataset import build
from erc_pose.data.pseudo_label import held_out_times, is_near_held_out
from erc_pose.data.split import DEFAULT_SESSION_GAP_S
from erc_pose.evaluate import evaluate
from erc_pose.train import train



def task_of(config):
    """Dataset task of a config name: pose_sym -> pose, obb_s -> obb, baseline -> detect."""
    return next((task for task in ("pose", "obb") if config.startswith(task)), "detect")


def filtered_extra(extra_dir, fold_dataset, out_dir, gap_s=DEFAULT_SESSION_GAP_S):
    """Copy of a YOLO Pose folder without the frames recorded near this fold's val/test frames."""
    times, _ = held_out_times(fold_dataset)
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    kept = 0
    for image in Path(extra_dir).glob("images/*"):
        if is_near_held_out(image.stem, times, gap_s):
            continue
        for src, dst in ((image, out_dir / "images" / image.name),
                         (Path(extra_dir) / "labels" / f"{image.stem}.txt", out_dir / "labels" / f"{image.stem}.txt")):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        kept += 1
    return out_dir, kept


def summarise(rows):
    """Mean and standard deviation of every numeric metric."""
    values = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
                values[key].append(value)
    return {key: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} for key, v in values.items()}


def write_summary(results_root):
    """Mean ± std per model over the fold result files; writes summary.json and summary.md."""
    results_root = Path(results_root)
    by_model = defaultdict(list)
    for path in sorted(results_root.glob("*_f*.json")):
        by_model[path.stem.rsplit("_f", 1)[0]].append(json.loads(path.read_text()))
    summary = {model: summarise(rows) for model, rows in by_model.items()}
    (results_root / "summary.json").write_text(json.dumps(summary, indent=2))
    keys = ["map50", "map50_95", "recall", "kpt_mae_px", "angle_axis_mae_deg", "angle_axis_within_5deg",
            "angle_mae_deg", "angle_within_5deg", "direction_acc"]
    lines = ["| model | folds | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 2)]
    for model, stats in summary.items():
        cells = [f"{stats[k]['mean']:.3f} ± {stats[k]['std']:.3f}" if k in stats else "–" for k in keys]
        lines.append(f"| {model} | {len(by_model[model])} | " + " | ".join(cells) + " |")
    (results_root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return summary


def mix(axis_name, direction_name, folds, name, tta=False):
    """Evaluate each fold's OBB from one experiment directed by the pose model of another."""
    results_root = Path("results/cv") / name
    for fold in range(folds):
        fold_dir = Path("data/cv") / direction_name / f"fold{fold}"  # same human-labelled split in both
        evaluate(Path(f"runs/cv/{axis_name}_obb_f{fold}/weights/best.pt"), fold_dir / "obb" / "data.yaml",
                 f"{axis_name}_obb+{direction_name}_pose_f{fold}", gt_dir=fold_dir / "pose", out_dir=results_root,
                 fps=False, direction_weights=Path(f"runs/cv/{direction_name}_pose_f{fold}/weights/best.pt"),
                 direction_tta=tta)
    return write_summary(results_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", help="YOLO Pose export with the human labels")
    parser.add_argument("--configs", nargs="+", default=["pose", "obb"])
    parser.add_argument("--name", required=True, help="experiment name")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--only-folds", type=int, nargs="*", help="run a subset of the folds")
    parser.add_argument("--extra-train", help="YOLO Pose folder (e.g. pseudo-labels) added to every train split")
    parser.add_argument("--boxes", nargs="+", default=[], help="YOLO Detection export(s) with drawn boxes")
    parser.add_argument("--predicted-boxes", nargs="+", default=[],
                        help="model boxes (e.g. for pseudo-labelled frames); they only attach to existing lines")
    parser.add_argument("--kpt-order", choices=("labelled", "left-to-right"), default="labelled")
    parser.add_argument("--fuse", action="store_true", help="also evaluate obb with direction from pose")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="training overrides")
    parser.add_argument("--weights-file", default="best.pt", choices=("best.pt", "last.pt"),
                        help="checkpoint to evaluate; best.pt is chosen by box/keypoint mAP, which can pick an "
                             "early epoch whose keypoints have not spread to the probe ends yet")
    parser.add_argument("--mix", nargs=2, metavar=("AXIS_EXP", "DIRECTION_EXP"),
                        help="only evaluate OBB models of one finished experiment directed by pose models of another")
    parser.add_argument("--tta", action="store_true", help="with --mix: flip test-time augmentation for the direction")
    args = parser.parse_args()
    if args.mix:
        mix(*args.mix, args.folds, args.name, args.tta)
        return
    if not args.export:
        parser.error("--export is required")

    import yaml

    overrides = {k: yaml.safe_load(v) for k, _, v in (item.partition("=") for item in args.set)}
    data_root, results_root = Path("data/cv") / args.name, Path("results/cv") / args.name
    for fold in args.only_folds if args.only_folds else range(args.folds):
        fold_dir = data_root / f"fold{fold}"
        build(yolo_dirs=[args.export], out_dir=fold_dir, fold=fold, folds=args.folds, kpt_order=args.kpt_order,
              box_dirs=args.boxes, predicted_box_dirs=args.predicted_boxes)
        if args.extra_train:
            extra, kept = filtered_extra(args.extra_train, fold_dir / "pose", data_root / f"extra{fold}")
            print(f"fold {fold}: {kept} extra training frames")
            build(yolo_dirs=[args.export], out_dir=fold_dir, fold=fold, folds=args.folds, extra_train=[extra],
                  kpt_order=args.kpt_order, box_dirs=args.boxes, predicted_box_dirs=args.predicted_boxes)
        weights = {}
        for config in args.configs:
            task = task_of(config)
            data = fold_dir / task / "data.yaml"
            run = f"{args.name}_{config}_f{fold}"
            model = train(f"configs/{config}.yaml", {**overrides, "name": run, "exist_ok": True,
                                                     "project": str(Path("runs/cv").resolve()), "plots": False},
                          data=data)
            weights[config] = Path(model.trainer.save_dir) / "weights" / args.weights_file
            evaluate(weights[config], data, f"{config}_f{fold}", gt_dir=fold_dir / "pose",
                     out_dir=results_root, fps=False)
        if args.fuse and "obb" in weights and "pose" in weights:
            evaluate(weights["obb"], fold_dir / "obb" / "data.yaml", f"obb+pose_f{fold}", gt_dir=fold_dir / "pose",
                     out_dir=results_root, fps=False, direction_weights=weights["pose"])

    write_summary(results_root)

if __name__ == "__main__":
    main()
