from geopy.geocoders import Nominatim
import numpy as np

def get_lat_long(city_name):
    geolocator = Nominatim(user_agent="geoapi", timeout=5)
    location = geolocator.geocode(city_name)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None
        
def haversine(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points (km)."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def calculate_duration(distance):
    if distance > 3000:
        speed = 830
    else:
        speed = 850
    
    return distance / speed

def same_country(city1, city2):
    lat1,long1 = get_lat_long(city1)
    lat2,long2 = get_lat_long(city2)
    geolocator = Nominatim(user_agent="geoapi", timeout=5)
    
    location1 = geolocator.reverse((lat1, long1), addressdetails=True)
    location2 = geolocator.reverse((lat2, long2), addressdetails=True)
    if (location1 and location1.raw.get("address")) and (location2 and location2.raw.get("address")):
        country1= location1.raw["address"]['country_code']
        country2= location2.raw["address"]['country_code']
        return country1 == country2