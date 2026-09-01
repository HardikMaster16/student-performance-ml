from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


sample_student = {
    "school": "GP",
    "sex": "F",
    "age": 17,
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "A",
    "Medu": 4,
    "Fedu": 4,
    "Mjob": "teacher",
    "Fjob": "health",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 1,
    "studytime": 3,
    "failures": 0,
    "schoolsup": "yes",
    "famsup": "yes",
    "paid": "no",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 3,
    "Dalc": 1,
    "Walc": 1,
    "health": 5,
    "absences": 2
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json=sample_student
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "label" in data
    assert "at_risk_probability" in data

    assert data["prediction"] in [0, 1]
    assert data["label"] in ["Good", "At Risk"]

    assert 0 <= data["at_risk_probability"] <= 1


def test_invalid_age():
    invalid_student = sample_student.copy()

    invalid_student["age"] = "hello"

    response = client.post(
        "/predict",
        json=invalid_student
    )

    assert response.status_code == 422


def test_missing_required_field():
    incomplete_student = sample_student.copy()

    del incomplete_student["studytime"]

    response = client.post(
        "/predict",
        json=incomplete_student
    )

    assert response.status_code == 422


def test_prediction_label_consistency():
    response = client.post(
        "/predict",
        json=sample_student
    )

    assert response.status_code == 200

    data = response.json()

    if data["prediction"] == 1:
        assert data["label"] == "At Risk"
    else:
        assert data["label"] == "Good"