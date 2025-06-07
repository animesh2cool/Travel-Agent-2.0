from serpapi import GoogleSearch

params = {
  "api_key": "ab9d62a4b8a76bf6e061b8aa07b5c7f024a978660f6d34f874a6715fa7e704e8", # Your SerpApi API Key
  "engine": "Google Flights", # Specifies to use the Google Flights engine
  "hl": "en", # Host Language: English
  "gl": "us", # Geo Location: United States (search will be performed as if from the US)
  "departure_id": "CDG", # Departure Airport IATA Code: Paris Charles de Gaulle Airport
  "arrival_id": "AUS", # Arrival Airport IATA Code: Austin-Bergstrom International Airport
  "outbound_date": "2025-06-06", # Outbound Flight Date: June 6, 2025
  "return_date": "2025-06-12", # Return Flight Date: June 12, 2025
  "currency": "USD" # Currency for prices: United States Dollar
}

search = GoogleSearch(params)
results = search.get_dict()

print("Flight Search Results:", results)