import argparse
import json
import cv2
from pathlib import Path
from .pipeline import registry, run


def main():
    parser = argparse.ArgumentParser(description="Authorized synthetic-media research lab")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("capabilities")
    process = sub.add_parser("prepare")
    process.add_argument("reference", type=Path)
    process.add_argument("--output", type=Path, default=Path(".lab/jobs"))
    process.add_argument("--authorized", action="store_true")
    process.add_argument("--license-note", required=True)
    swap = sub.add_parser("swap", parents=[], help="Landmark-based image face swap")
    swap.add_argument("reference", type=Path)
    swap.add_argument("target", type=Path)
    swap.add_argument("--allow-low-detail", action="store_true", help="Experimental processing; results remain visually unverified")
    swap.add_argument("--backend", choices=["auto", "neural", "baseline", "hyperswap", "best", "facefusion", "remote"], default="auto")
    swap.add_argument("--output", type=Path, default=Path(".lab/jobs"))
    swap.add_argument("--authorized", action="store_true")
    swap.add_argument("--license-note", required=True, help="Authorization covering both reference and target")
    args = parser.parse_args()
    try:
        result = registry().capabilities() if args.command == "capabilities" else run(args.reference, args.output, args.authorized, args.license_note,
                  stage="face_swapping" if args.command == "swap" else "face_processing",
                  target=args.target if args.command == "swap" else None,
                  backend=args.backend if args.command == "swap" else "auto",
                  allow_low_detail=args.allow_low_detail if args.command == "swap" else False)
    except (ValueError, OSError, RuntimeError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
