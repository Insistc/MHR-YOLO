"""Single-device training entry point for MHR-YOLO.

Applies cosine temperature annealing to DP-MoE blocks, 10-epoch linear warmup
to the CB-RPL detection head, and records auxiliary routing/prototype metrics.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def set_module_epoch(trainer):
    from ultralytics.nn.modules.ScaleAwareMoE_Block import DP_MoE
    from ultralytics.nn.modules.head import CBRPLDetect
    from ultralytics.utils.torch_utils import unwrap_model

    for module in unwrap_model(trainer.model).modules():
        if isinstance(module, (DP_MoE, CBRPLDetect)):
            module.set_epoch(trainer.epoch, trainer.epochs)


def write_aux_metrics(trainer):
    from ultralytics.utils.torch_utils import unwrap_model

    criterion = getattr(unwrap_model(trainer.model), "criterion", None)
    if criterion is not None and hasattr(criterion, "consume_aux_epoch"):
        record = {"epoch": int(trainer.epoch) + 1, **criterion.consume_aux_epoch()}
        with (Path(trainer.save_dir) / "auxiliary_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path, help='Exact dataset YAML')
    parser.add_argument('--model', type=Path, default=ROOT/'configs/models/mhr-yolo11n-fcidd.yaml')
    parser.add_argument('--device', default='0', help='One GPU index or cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--project', type=Path, default=ROOT/'runs/train')
    parser.add_argument('--name', default=None)
    parser.add_argument('--epochs', type=int, default=None, help='Override only for smoke tests or new experiments')
    parser.add_argument('--batch', type=int, default=None)
    args = parser.parse_args()
    if ',' in args.device:
        parser.error('This entry point supports a single device only.')
    import yaml
    import ultralytics
    from ultralytics import YOLO
    assert Path(ultralytics.__file__).resolve().is_relative_to(ROOT), 'Bundled runtime was not imported'
    config = yaml.safe_load((ROOT/'configs/train.yaml').read_text())
    config.update(data=str(args.data.resolve(strict=True)), device=args.device, seed=args.seed,
                  project=str(args.project.resolve()), name=args.name or f'mhr-yolo-seed{args.seed}')
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch is not None:
        config['batch'] = args.batch
    model = YOLO(str(args.model.resolve(strict=True)))
    model.add_callback('on_train_epoch_start', set_module_epoch)
    model.add_callback('on_train_epoch_end', write_aux_metrics)
    model.train(**config)


if __name__ == '__main__':
    main()
