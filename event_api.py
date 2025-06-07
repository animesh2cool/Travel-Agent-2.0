import requests
from datetime import datetime, timedelta

API_KEY = "KoA4C8W6pYC8KhJ-VLRXeEA_LhMxcXLFMqXe6L4n"

# Define the date range for the search
START_DATE = "2025-05-23"
END_DATE = "2026-06-02"

# London, UK coordinates (approximate center)
LONDON_LATITUDE = 51.5074
LONDON_LONGITUDE = -0.1278

# --- API Call ---
response = requests.get(
    url="https://api.predicthq.com/v1/events/",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json"
    },
    params={
        # Geographic filtering: events within 50km of London
        "within": f"50km@{LONDON_LATITUDE},{LONDON_LONGITUDE}",

        # Date range for events
        "active.gte": START_DATE,
        "active.lte": END_DATE,
        "active.tz": "Europe/London", # Specify London's timezone for date parameters

        # Categories relevant for holiday travel in London
        "category": "festivals,performing-arts,public-holidays,concerts,community,sports",

        # Optional: Filter by expected impact (Major/Significant local rank)
        "local_rank_level": "4,5",

        # Optional: Sort by start date then by PHQ rank
        "sort": "start,-rank",

        # Optional: Limit the number of results per page
        "limit": 20,

        # Optional: Exclude potentially brand-unsafe events
        "brand_unsafe.exclude": "true",
    }
)

# --- Process and Print Results ---
if response.status_code == 200:
    data = response.json()
    print(f"Successfully retrieved Active Events for London between {START_DATE} and {END_DATE}:")
    for event in data.get("results", []):
        print(f"  Title: {event.get('title')}")
        print(f"  Category: {event.get('category')}")
        print(f"  Start: {event.get('start')}")
        print(f"  End: {event.get('end')}")
        print(f"  PHQ Rank: {event.get('phq_rank')}")
        print(f"  Local Rank: {event.get('local_rank')}")

        # --- FIX FOR THE 'location_info' ERROR ---
        location_info = event.get('location')
        if location_info:
            if isinstance(location_info, list) and location_info:
                # If it's a list, take the first location object
                first_location = location_info[0]
                if isinstance(first_location, dict):
                    print(f"  Location: Lat {first_location.get('latitude')}, Lng {first_location.get('longitude')}")
                else:
                    print("  Location: Format unexpected (list element not dict)")
            elif isinstance(location_info, dict):
                # If it's a single dictionary, process directly
                print(f"  Location: Lat {location_info.get('latitude')}, Lng {location_info.get('longitude')}")
            else:
                print("  Location: Not specified or unknown format")
        else:
            print("  Location: Not specified (broad area event)")
        # --- END FIX ---

        print(f"  Description: {event.get('description', 'N/A')}")
        print("-" * 30)

    if not data.get("results"):
        print("No active events found for the specified criteria in London.")

else:
    print(f"Error: {response.status_code} - {response.text}")
