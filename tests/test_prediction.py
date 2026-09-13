"""
Unit tests for the inference engine and patient input validator.
"""

import pytest
from src.validation import validate_patient_input, ValidationError
from src.prediction import predict_patient
from config.config import TARGET_CLASSES, MEDICAL_DISCLAIMER


VALID_PATIENT = {
    "Age": 52.0,
    "Sex": "F",
    "Ascites": 0.0,
    "Hepatomegaly": 1.0,
    "Spiders": 0.0,
    "Edema": 0.0,
    "Bilirubin": 2.5,
    "Cholesterol": 300.0,
    "Albumin": 3.4,
    "Copper": 80.0,
    "Alk_Phos": 1700.0,
    "SGOT": 110.0,
    "Triglycerides": 125.0,
    "Platelets": 220.0,
    "Prothrombin": 10.7,
    "Stage": 3.0,
}


def test_validate_patient_input_accepts_valid_data():
    is_valid, err = validate_patient_input(VALID_PATIENT)
    assert is_valid is True
    assert err is None


def test_validate_patient_input_rejects_negative_age():
    bad_patient = VALID_PATIENT.copy()
    bad_patient["Age"] = -5.0
    is_valid, err = validate_patient_input(bad_patient)
    assert is_valid is False
    assert "outside physiologically plausible range" in err


def test_validate_patient_input_rejects_extreme_bilirubin():
    bad_patient = VALID_PATIENT.copy()
    bad_patient["Bilirubin"] = 250.0  # Impossibly high (> 100)
    is_valid, err = validate_patient_input(bad_patient)
    assert is_valid is False
    assert "Bilirubin" in err


def test_validate_patient_input_rejects_invalid_categorical():
    bad_patient = VALID_PATIENT.copy()
    bad_patient["Sex"] = "UnknownGender"
    is_valid, err = validate_patient_input(bad_patient)
    assert is_valid is False
    assert "Sex" in err


def test_predict_patient_raises_validation_error_on_bad_input():
    bad_patient = VALID_PATIENT.copy()
    bad_patient["Stage"] = 10.0  # Histologic stage is 1-4
    with pytest.raises(ValidationError):
        predict_patient(bad_patient)


def test_predict_patient_returns_canonical_schema():
    result = predict_patient(VALID_PATIENT)

    assert "predicted_class" in result
    assert result["predicted_class"] in [0, 1, 2]

    assert "predicted_label" in result
    assert result["predicted_label"] in list(TARGET_CLASSES.values())

    assert "probabilities" in result
    probs = result["probabilities"]
    assert set(probs.keys()) == set(TARGET_CLASSES.values())
    prob_sum = sum(probs.values())
    assert 0.99 <= prob_sum <= 1.01

    assert "warning" in result
    assert result["warning"] == MEDICAL_DISCLAIMER
