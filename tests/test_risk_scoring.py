import numpy as np
import pytest

from src.risk_scoring import risk_bands


def test_exact_boundaries():
    inputs = [0, np.nextafter(.30, 0), .30, .60, np.nextafter(.60, 1), 1]
    assert list(risk_bands(inputs)) == [
        "Low Risk", "Low Risk", "Medium Risk", "Medium Risk", "High Risk", "High Risk",
    ]
    assert list(risk_bands([.25, .5, .51], lower=.25, upper=.5)) == [
        "Medium Risk", "Medium Risk", "High Risk",
    ]


@pytest.mark.parametrize("probs,bounds", [([.3], (.8, .6)), ([.3], (.3, .3)), ([float("nan")], (.3, .6)),
                                         ([1.1], (.3, .6)), ([-.1], (.3, .6))])
def test_invalid_boundaries_and_probabilities(probs, bounds):
    with pytest.raises(ValueError):
        risk_bands(probs, *bounds)
