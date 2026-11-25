"""Evaluate a trained MHR-YOLO checkpoint on a fixed validation or test split."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()
    import ultralytics
    from ultralytics import YOLO
    assert Path(ultralytics.__file__).resolve().is_relative_to(ROOT), 'Bundled runtime was not imported'
    model = YOLO(str(args.weights.resolve(strict=True)))
    results = model.val(data=str(args.data.resolve(strict=True)), split=args.split,
                        imgsz=640, batch=args.batch, device=args.device,
                        iou=0.7, max_det=300, half=False, augment=False, plots=False,
                        project=str(ROOT/'runs/eval'), name=args.split)
    print(json.dumps({k: float(v) for k, v in results.results_dict.items()}, indent=2))


if __name__ == '__main__':
    main()
