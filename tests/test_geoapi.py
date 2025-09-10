# tests/test_geoapi.py
import pytest
from src.utils import get_lat_long

def test_geoapi_availability():
    """Check if geo API responds with valid coordinates."""
    city = "Delhi, India"
    lat, lon = get_lat_long(city)

    assert lat is not None, "GeoAPI failed: Latitude missing"
    assert lon is not None, "GeoAPI failed: Longitude missing"
    assert isinstance(lat, (float, int)), "Latitude is not numeric"
    assert isinstance(lon, (float, int)), "Longitude is not numeric"
