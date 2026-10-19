"""BRIDGE command line; import heavy dependencies only for the chosen command."""

import argparse
import json
from pathlib import Path

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bridge-st", description="BRIDGE spatial expression training, prediction, and evaluation")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("make-demo", help="Generate synthetic labeled and label-free example data")
    demo.add_argument("--out", type=Path, required=True)
    demo.add_argument("--seed", type=int, default=7)
    validate = sub.add_parser("validate", help="Validate an NPZ data bundle")
    validate.add_argument("--data", type=Path, required=True)
    validate.add_argument("--targets", action="store_true")
    train = sub.add_parser("train", help="Train using cell-level expression labels on training spots")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument("--epochs", type=int)
    train.add_argument("--threads", type=int, default=1)
    pred = sub.add_parser("predict", help="Predict from observations and morphology without cell-expression labels")
    pred.add_argument("--data", type=Path, required=True)
    pred.add_argument("--checkpoint", type=Path, required=True)
    pred.add_argument("--out", type=Path, required=True)
    pred.add_argument("--device", default="cpu")
    pred.add_argument("--split", choices=["all", "test"], default="all")
    pred.add_argument("--threads", type=int, default=1)
    evaluate = sub.add_parser("evaluate", help="Evaluate saved predictions with explicit split and metric definitions")
    evaluate.add_argument("--reference", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--split", choices=["all", "test"], default="test")
    evaluate.add_argument("--threshold", type=float, default=.3)
    evaluate.add_argument("--mmd-bandwidth", type=float, default=1.)
    evaluate.add_argument("--mmd-max-samples", type=int, default=1024)
    prepare = sub.add_parser("prepare-legacy", help="Convert the documented original HD research input layout")
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--genes", type=Path)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--trust-features-pickle", action="store_true")
    prepare.add_argument("--max-spots", type=int)
    prepare.add_argument("--gene-order", choices=["legacy", "panel"], default="legacy")
    convert = sub.add_parser("convert-checkpoint", help="Import tensor state dicts with an explicit training gene panel")
    convert.add_argument("--source", type=Path, required=True)
    convert.add_argument("--genes", type=Path, required=True)
    convert.add_argument("--config", type=Path, required=True)
    convert.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "make-demo":
            from .data import make_demo
            result = make_demo(args.out, args.seed)
        elif args.command == "validate":
            from .data import load_bundle
            data = load_bundle(args.data, targets=args.targets)
            result = {"cells": len(data.cell_ids), "spots": len(data.spot_ids), "genes": len(data.genes), "image_feature_width": data.image_features.shape[1], "targets_loaded": data.cell_expression is not None}
        elif args.command in {"train", "convert-checkpoint"}:
            from .network import Config
            values = json.loads(args.config.read_text())
            if args.command == "train" and args.epochs is not None:
                values["epochs"] = args.epochs
            config = Config.from_dict(values)
            if args.command == "train":
                import torch
                from .engine import train
                if args.threads < 1:
                    raise ValueError("threads must be positive")
                torch.set_num_threads(args.threads)
                result = train(args.data, args.out, config, args.device)
            else:
                from .engine import convert_checkpoint
                result = convert_checkpoint(args.source, args.genes, config, args.out)
        elif args.command == "predict":
            import torch
            from .engine import predict
            if args.threads < 1:
                raise ValueError("threads must be positive")
            torch.set_num_threads(args.threads)
            result = predict(args.data, args.checkpoint, args.out, args.device, args.split)
        elif args.command == "evaluate":
            from .metrics import evaluate
            result = evaluate(args.reference, args.predictions, args.out, args.split, args.threshold, args.mmd_bandwidth, args.mmd_max_samples)
        else:
            from .prepare import prepare_legacy
            result = prepare_legacy(args.root, args.out, args.genes, args.trust_features_pickle, args.max_spots, args.gene_order)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, f"bridge-st: {exc}\n")
    print(json.dumps(result) if isinstance(result, dict) else str(result))
