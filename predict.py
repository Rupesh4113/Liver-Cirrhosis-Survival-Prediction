"""
Command-Line and programmatic prediction script for Liver Cirrhosis Survival Prediction.

Usage:
    python predict.py --sample
    python predict.py --json '{"Age": 55, "Bilirubin": 2.5, "Albumin": 3.4, ...}'
    python predict.py --file patient.json
"""

import sys
import json
import argparse
from pathlib import Path

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.prediction import predict_patient
from src.validation import ValidationError


SAMPLE_PATIENT = {
    "Age": 56.0,
    "Sex": "F",
    "Ascites": 0.0,
    "Hepatomegaly": 1.0,
    "Spiders": 0.0,
    "Edema": 0.0,
    "Bilirubin": 2.8,
    "Cholesterol": 310.0,
    "Albumin": 3.48,
    "Copper": 85.0,
    "Alk_Phos": 1650.0,
    "SGOT": 110.0,
    "Triglycerides": 115.0,
    "Platelets": 210.0,
    "Prothrombin": 10.8,
    "Stage": 3.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict patient survival outcome class.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", action="store_true", help="Run prediction on a canonical sample patient.")
    group.add_argument("--json", type=str, help="Raw JSON string containing patient biomarker values.")
    group.add_argument("--file", type=str, help="Path to JSON file containing patient biomarker dictionary.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.sample:
        patient_data = SAMPLE_PATIENT
        print("\nUsing Canonical Sample Patient Profile:")
        print(json.dumps(patient_data, indent=2))
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File '{file_path}' does not exist.", file=sys.stderr)
            sys.exit(1)
        with open(file_path, "r", encoding="utf-8") as f:
            patient_data = json.load(f)
    elif args.json:
        try:
            patient_data = json.loads(args.json)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON string: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        result = predict_patient(patient_data)
        print("\n" + "=" * 60)
        print("PREDICTION RESULT (Research / Educational)")
        print("=" * 60)
        print(f"Predicted Class ID:    {result['predicted_class']}")
        print(f"Predicted Class Label: {result['predicted_label']}")
        print("\nClass Probabilities:")
        for cls_name, prob in result["probabilities"].items():
            bar = "#" * int(prob * 30)
            print(f"  - {cls_name:<12}: {prob * 100:>6.2f}% | {bar}")
        print(f"\nModel: {result['model_version']}")
        print("-" * 60)
        print(f"DISCLAIMER: {result['warning']}")
        print("=" * 60 + "\n")
    except ValidationError as val_err:
        print(f"\n[Validation Error] {val_err}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[Execution Error] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
