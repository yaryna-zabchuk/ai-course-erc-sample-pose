"""Train a detection, pose or OBB model from a config file.

    python -m erc_pose.train configs/pose.yaml
    python -m erc_pose.train configs/pose.yaml --set epochs=5 device=cpu
"""

import argparse
from pathlib import Path

import yaml


def noise_augmentations():
    """Pixel-level Albumentations transforms: sensor noise, exposure and blur.

    Only pixel-level transforms belong here, because Ultralytics does not pass
    keypoints or oriented boxes through Albumentations.
    """
    import albumentations as A

    return [
        A.GaussNoise(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.RandomGamma(p=0.2),
        A.MotionBlur(blur_limit=5, p=0.1),
        A.ImageCompression(p=0.2),
    ]


def train(config_path, overrides=None, data=None):
    from ultralytics import YOLO

    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    args = dict(config.get("train") or {})
    args.update(overrides or {})
    args.setdefault("project", str(Path("runs").resolve()))
    args.setdefault("name", config_path.stem)
    if config.get("noise_aug"):
        args["augmentations"] = noise_augmentations()
    if config.get("symmetric_keypoints"):
        from erc_pose import symmetric

        symmetric.enable()

    model = YOLO(config["model"])
    model.train(data=str(Path(data or config["data"]).resolve()), **args)
    print(f"best weights: {Path(model.trainer.save_dir) / 'weights' / 'best.pt'}")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--data", help="dataset yaml instead of the one in the config")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override training arguments, e.g. epochs=5 device=cpu")
    args = parser.parse_args()
    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        overrides[key] = yaml.safe_load(value)
    train(args.config, overrides, args.data)


if __name__ == "__main__":
    main()
