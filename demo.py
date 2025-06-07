import os
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
import httpx
from dotenv import load_dotenv
from litellm import completion
from serpapi import GoogleSearch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Travel Planner with Weather", version="7.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class TravelRequest(BaseModel):
    from_city: str
    destination: str
    start_date: str
    end_date: str
    interests: str
    budget: str = "medium"

class TravelResponse(BaseModel):
    itinerary: dict
    guide: dict
    logistics: dict
    weather: dict
    language: str
    flights: Optional[List[Dict[str, Any]]] = None

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"}
    )

# Weather Service
class WeatherService:
    def __init__(self):
        self.api_key = os.getenv("WEATHER_API_KEY")
        self.base_url = "http://api.openweathermap.org/data/2.5"
        
    async def get_weather_forecast(self, city: str, days: int = 5) -> Dict[Any, Any]:
        """Get weather forecast for a city"""
        try:
            # If no API key, return mock data
            if not self.api_key:
                logger.warning("No weather API key found, returning mock weather data")
                return self._get_mock_weather_data(city, days)
            
            # Get current weather
            current_url = f"{self.base_url}/weather"
            current_params = {
                "q": city,
                "appid": self.api_key,
                "units": "metric"
            }
            
            # Get forecast
            forecast_url = f"{self.base_url}/forecast"
            forecast_params = {
                "q": city,
                "appid": self.api_key,
                "units": "metric",
                "cnt": min(days * 8, 40)  # 8 forecasts per day, max 40
            }
            
            async with httpx.AsyncClient() as client:
                current_response = await client.get(current_url, params=current_params)
                forecast_response = await client.get(forecast_url, params=forecast_params)
                
                if current_response.status_code != 200 or forecast_response.status_code != 200:
                    logger.error(f"Weather API error: {current_response.status_code}, {forecast_response.status_code}")
                    return self._get_mock_weather_data(city, days)
                
                current_data = current_response.json()
                forecast_data = forecast_response.json()
                
                return self._format_weather_data(current_data, forecast_data, city)
                
        except Exception as e:
            logger.error(f"Weather service error: {str(e)}")
            return self._get_mock_weather_data(city, days)
    
    def _format_weather_data(self, current: dict, forecast: dict, city: str) -> Dict[Any, Any]:
        """Format weather data for travel planning"""
        try:
            formatted_forecast = []
            daily_data = {}
            
            # Process forecast data (group by day)
            for item in forecast.get("list", []):
                date = datetime.fromtimestamp(item["dt"]).strftime("%Y-%m-%d")
                if date not in daily_data:
                    daily_data[date] = {
                        "date": date,
                        "temps": [],
                        "conditions": [],
                        "humidity": [],
                        "wind_speed": []
                    }
                
                daily_data[date]["temps"].append(item["main"]["temp"])
                daily_data[date]["conditions"].append(item["weather"][0]["description"])
                daily_data[date]["humidity"].append(item["main"]["humidity"])
                daily_data[date]["wind_speed"].append(item["wind"]["speed"])
            
            # Create daily summaries
            for date, data in daily_data.items():
                formatted_forecast.append({
                    "date": date,
                    "high_temp": round(max(data["temps"])),
                    "low_temp": round(min(data["temps"])),
                    "condition": max(set(data["conditions"]), key=data["conditions"].count),
                    "humidity": round(sum(data["humidity"]) / len(data["humidity"])),
                    "wind_speed": round(sum(data["wind_speed"]) / len(data["wind_speed"]), 1)
                })
            
            return {
                "city": city,
                "current": {
                    "temperature": round(current["main"]["temp"]),
                    "condition": current["weather"][0]["description"],
                    "humidity": current["main"]["humidity"],
                    "wind_speed": round(current["wind"]["speed"], 1)
                },
                "forecast": formatted_forecast[:7],  # Limit to 7 days
                "travel_advice": self._generate_travel_advice(formatted_forecast[:7])
            }
        except Exception as e:
            logger.error(f"Weather formatting error: {str(e)}")
            return self._get_mock_weather_data(city, 7)
    
    def _generate_travel_advice(self, forecast: list) -> str:
        """Generate travel advice based on weather"""
        if not forecast:
            return "Weather data unavailable for travel planning."
        
        advice = []
        avg_temp = sum(day["high_temp"] for day in forecast) / len(forecast)
        
        if avg_temp > 30:
            advice.append("Pack light, breathable clothing and sun protection.")
        elif avg_temp > 20:
            advice.append("Comfortable weather expected. Pack light layers.")
        elif avg_temp > 10:
            advice.append("Pack warm clothing and layers.")
        else:
            advice.append("Pack heavy winter clothing.")
        
        rainy_days = sum(1 for day in forecast if "rain" in day["condition"].lower())
        if rainy_days > len(forecast) / 2:
            advice.append("Frequent rain expected - pack umbrella and waterproof gear.")
        
        return " ".join(advice)
    
    def _get_mock_weather_data(self, city: str, days: int) -> Dict[Any, Any]:
        """Return mock weather data when API is unavailable"""
        return {
            "city": city,
            "current": {
                "temperature": 22,
                "condition": "partly cloudy",
                "humidity": 65,
                "wind_speed": 3.2
            },
            "forecast": [
                {
                    "date": (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + 
                            timedelta(days=i)).strftime("%Y-%m-%d"),
                    "high_temp": 25 + (i % 3),
                    "low_temp": 18 + (i % 2),
                    "condition": ["sunny", "partly cloudy", "cloudy"][i % 3],
                    "humidity": 60 + (i * 2),
                    "wind_speed": 3.0 + (i * 0.5)
                } for i in range(min(days, 7))
            ],
            "travel_advice": "Mock weather data - comfortable conditions expected for travel."
        }

# Flight Service
class FlightService:
    def __init__(self):
        self.api_key = os.getenv("SERPAPI_KEY")

    def get_flights(self, from_airport: str, to_airport: str, date: str, currency="USD", travel_class=1):
        try:
            if not self.api_key:
                raise ValueError("SERPAPI_KEY not found in environment variables")

            params = {
                "api_key": self.api_key,
                "engine": "Google Flights",
                "type": 2,
                "hl": "en",
                "gl": "us",
                "departure_id": from_airport,
                "arrival_id": to_airport,
                "outbound_date": date,
                "currency": currency,
                "travel_class": travel_class,
                "sort_by": 2
            }

            search = GoogleSearch(params)
            results = search.get_dict()
            return results.get("flights_results", [])  # Adjust this based on actual response
        except Exception as e:
            logger.error(f"Flight API error: {e}")
            return []


# Custom OpenRouter LLM for CrewAI
class OpenRouterLLM(LLM):
    def __init__(self, model: str, api_key: str, base_url: str, temperature: float = 0.7):
        super().__init__(model=model)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
    
    def _call(self, prompt: str, **kwargs) -> str:
        try:
            # Add required headers for OpenRouter
            headers = {
                "HTTP-Referer": "http://localhost:8000",  # Your app's domain
                "X-Title": "Travel Planner App"  # Your app's name
            }
            
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.api_key,
                api_base=self.base_url,
                temperature=self.temperature,
                headers=headers,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LiteLLM call failed: {str(e)}")
            raise

def get_llm():
    """Initialize OpenRouter LLM for CrewAI"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")
    
    return OpenRouterLLM(
        # Use a more reliable model
        model="openrouter/meta-llama/llama-4-maverick:free",  # Changed from llama-4-maverick
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",  # Updated base URL
        temperature=0.7
    )

# Update the test connection function as well
def test_openrouter_connection(api_key: str) -> bool:
    """Test if OpenRouter API is available"""
    try:
        headers = {
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Travel Planner App"
        }
        
        response = completion(
            model="openai/gpt-3.5-turbo",  # Changed model
            messages=[{"role": "user", "content": "Hello"}],
            api_key=api_key,
            api_base="https://openrouter.ai/api/v1",  # Updated base URL
            headers=headers,
            max_tokens=10
        )
        return True
    except Exception as e:
        logger.error(f"OpenRouter connection test failed: {str(e)}")
        return False

# Agent Creation
def create_agents(llm, weather_context: str = ""):
    try:
        weather_instruction = f"\nWeather Context for trip planning: {weather_context}" if weather_context else ""
        
        return (
            Agent(
                role="Local Guide Expert",
                goal="Find best attractions and activities considering weather conditions",
                backstory=f"You are a knowledgeable local guide with deep insights about city attractions, hidden gems, and the best activities. You always consider weather conditions when making recommendations.{weather_instruction}",
                llm=llm,
                verbose=True,
                allow_delegation=False
            ),
            Agent(
                role="Travel Logistics Expert", 
                goal="Research accommodation and transportation with weather considerations",
                backstory=f"You are an experienced travel coordinator who specializes in finding the best accommodations, transportation options, and practical travel information. You factor weather into all logistics planning.{weather_instruction}",
                llm=llm,
                verbose=True,
                allow_delegation=False
            ),
            Agent(
                role="Itinerary Planner",
                goal="Create weather-optimized travel plans",
                backstory=f"You are a professional travel designer who creates detailed, day-by-day itineraries. You excel at scheduling activities based on weather conditions and traveler interests.{weather_instruction}",
                llm=llm,
                verbose=True,
                allow_delegation=False
            )
        )
    except Exception as e:
        logger.error(f"Agent creation failed: {str(e)}")
        raise

# Main Endpoint
@app.post("/plan", response_model=TravelResponse)
async def create_plan(request: TravelRequest):
    try:
        # Check for required API key
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise HTTPException(500, detail="OPENROUTER_API_KEY not found in environment variables")

        # Initialize services
        weather_service = WeatherService()
        flight_service = FlightService()

        # Dates
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        end = datetime.strptime(request.end_date, "%Y-%m-%d")
        days = (end - start).days + 1

        # Get weather data
        weather_data = await weather_service.get_weather_forecast(request.destination, days)

        # Get flight data
        flights = flight_service.get_flights(
            from_airport=request.from_city,
            to_airport=request.destination,
            date=request.start_date
        )

        # Format flight info (for task prompt)
        flight_info = "\n".join([
            f"- {f.get('airline', 'N/A')} | {f.get('departure_time', 'N/A')} → {f.get('arrival_time', 'N/A')} | ${f.get('price', 'N/A')}"
            for f in flights[:3]
        ]) if flights else "No flight data available."

        # Create weather summary for LLM agents
        weather_summary = f"Current: {weather_data['current']['temperature']}°C, {weather_data['current']['condition']}"
        if weather_data['forecast']:
            temp_range = f"{min(d['low_temp'] for d in weather_data['forecast'])}-{max(d['high_temp'] for d in weather_data['forecast'])}°C"
            weather_summary += f" | Forecast: {temp_range} over {len(weather_data['forecast'])} days"
        weather_summary += f" | Advice: {weather_data['travel_advice']}"

        # Initialize LLM + agents
        llm = get_llm()
        guide, logistics, planner = create_agents(llm, weather_summary)

        # CrewAI Tasks
        logistics_task = Task(
            description=f"""Research logistics for a trip to {request.destination} from {request.start_date} to {request.end_date} with a {request.budget} budget.

Include:
- Accommodation options (hotel, hostel, Airbnb) within {request.budget} budget
- Transportation (flights, trains, public transit)
- Visa/documents needed
- Local customs and safety info
- Packing list for the weather

Weather Context: {weather_summary}

Available Flights:
{flight_info}

Be practical and specific.""",
            agent=logistics,
            expected_output="Detailed logistics report with travel and weather considerations"
        )

        guide_task = Task(
            description=f"""Find the best {request.interests} activities and attractions in {request.destination} for a {days}-day trip.

Include:
- Top attractions
- Local hidden gems
- Food and cultural experiences
- Indoor/outdoor options depending on weather

Weather Context: {weather_summary}

Deliver a rich, engaging guide based on the interests provided.""",
            agent=guide,
            expected_output="Guide with attractions and activities with weather suitability"
        )

        plan_task = Task(
            description=f"""Create a day-by-day itinerary for {request.destination} from {request.start_date} to {request.end_date}.

Include:
- Daily schedule of activities
- Weather-based adjustments (e.g., indoor on rainy days)
- Realistic travel time
- Meal suggestions and rest time
- Backup options

Use inputs from logistics and guide.

Weather Context: {weather_summary}""",
            agent=planner,
            context=[logistics_task, guide_task],
            expected_output="Detailed itinerary with timing, activities, and contingencies"
        )

        # Execute Crew
        crew = Crew(
            agents=[guide, logistics, planner],
            tasks=[logistics_task, guide_task, plan_task],
            process=Process.sequential,
            verbose=True
        )

        logger.info("Starting CrewAI planning sequence...")
        result = crew.kickoff()
        logger.info("CrewAI planning completed.")

        # Parse outputs
        def extract_task_result(task_result):
            if hasattr(task_result, 'raw'):
                return {"content": task_result.raw}
            elif hasattr(task_result, 'result'):
                return {"content": str(task_result.result)}
            elif isinstance(task_result, str):
                return {"content": task_result}
            else:
                return {"content": str(task_result)}

        if hasattr(result, 'tasks_output'):
            tasks_results = result.tasks_output
        elif isinstance(result, list):
            tasks_results = result
        else:
            tasks_results = [result] * 3

        logistics_result = extract_task_result(tasks_results[0]) if len(tasks_results) > 0 else {"content": "Logistics not available"}
        guide_result = extract_task_result(tasks_results[1]) if len(tasks_results) > 1 else {"content": "Guide not available"}
        itinerary_result = extract_task_result(tasks_results[2]) if len(tasks_results) > 2 else {"content": "Itinerary not available"}

        return TravelResponse(
            itinerary=itinerary_result,
            guide=guide_result,
            logistics=logistics_result,
            weather=weather_data,
            language="en",
            flights=flights
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Planning error: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=f"Failed to generate travel plan: {str(e)}")


# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not api_key:
            return {
                "status": "unhealthy",
                "openrouter": "no_api_key",
                "error": "OPENROUTER_API_KEY not found in environment variables"
            }
        
        # Test OpenRouter connection
        try:
            response = completion(
                model="openrouter/meta-llama/llama-4-maverick:free",
                messages=[{"role": "user", "content": "Say 'Hello' in one word"}],
                api_key=api_key,
                api_base="https://openrouter.ai/api/v1/chat/completions",
                max_tokens=10
            )
            
            return {
                "status": "healthy",
                "openrouter": "connected",
                "model": "meta-llama/llama-4-maverick:free",
                "test_response": response.choices[0].message.content.strip()[:50]
            }
        except Exception as api_error:
            return {
                "status": "unhealthy",
                "openrouter": "api_error",
                "error": str(api_error)
            }
            
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Get available models (OpenRouter specific)
@app.get("/models")
async def get_models():
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return {"error": "OPENROUTER_API_KEY not found"}
        
        # OpenRouter models endpoint
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
        
        if response.status_code == 200:
            models_data = response.json()
            return {
                "models": [model["id"] for model in models_data.get("data", [])],
                "count": len(models_data.get("data", []))
            }
        else:
            return {"error": f"Cannot fetch models from OpenRouter: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# Static files setup
PUBLIC_DIR = Path(__file__).parent / "public"
PUBLIC_DIR.mkdir(exist_ok=True)  # Create public directory if it doesn't exist

try:
    app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="static")
except RuntimeError:
    logger.warning("Public directory not found - static files not mounted")

if __name__ == "__main__":
    import uvicorn
    print("Starting Travel Planner with OpenRouter integration...")
    print("Make sure you have OPENROUTER_API_KEY in your environment variables")
    uvicorn.run(
        "demo:app",  # Use import string format
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )