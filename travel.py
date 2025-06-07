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

# Custom Ollama LLM for CrewAI
class OllamaLLM(LLM):
    def __init__(self, model: str = "ollama/llama3.2", base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model=model)  # Initialize parent class first
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = kwargs.get('temperature', 0.7)
    
    def _call(self, prompt: str, **kwargs) -> str:
        """Call Ollama API"""
        try:
            if not self.base_url:
                raise ValueError("Base URL cannot be None")
                
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model.replace("ollama/", ""),  # Remove ollama/ prefix if present
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": kwargs.get('max_tokens', 2048)
                }
            }
            
            response = requests.post(url, json=payload, timeout=120)
            
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                raise Exception(f"Ollama API error: {response.status_code}")
            
            result = response.json()
            return result.get("response", "No response generated")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama connection error: {str(e)}")
            raise Exception(f"Failed to connect to Ollama. Make sure Ollama is running: {str(e)}")
        except Exception as e:
            logger.error(f"Ollama call failed: {str(e)}")
            raise
    
    @property
    def _llm_type(self) -> str:
        return "ollama"

# LLM Configuration
def get_llm():
    """Initialize Ollama LLM for CrewAI"""
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model_name = "ollama/llama3.2"  # Simplified model name
    
    if not test_ollama_connection():
        raise ValueError("Ollama is not running. Please start with 'ollama serve'")
        
    return OllamaLLM(
        model=model_name,
        base_url=ollama_url,
        temperature=0.7
    )

# Test Ollama connection
def test_ollama_connection():
    """Test if Ollama is available"""
    try:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        response = requests.get(f"{ollama_url}/api/tags", timeout=10)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Ollama connection test failed: {str(e)}")
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
        # Check Ollama connection first
        if not test_ollama_connection():
            raise HTTPException(500, detail="Ollama is not running. Please start Ollama with 'ollama serve'")
        
        # Initialize services
        weather_service = WeatherService()
        
        # Get weather data
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        end = datetime.strptime(request.end_date, "%Y-%m-%d")
        days = (end - start).days + 1
        
        weather_data = await weather_service.get_weather_forecast(request.destination, days)
        
        # Create weather context for agents
        weather_summary = f"Current: {weather_data['current']['temperature']}°C, {weather_data['current']['condition']}"
        if weather_data['forecast']:
            temp_range = f"{min(d['low_temp'] for d in weather_data['forecast'])}-{max(d['high_temp'] for d in weather_data['forecast'])}°C"
            weather_summary += f" | Forecast: {temp_range} over {len(weather_data['forecast'])} days"
        weather_summary += f" | Advice: {weather_data['travel_advice']}"
        
        # Initialize LLM and agents
        llm = get_llm()
        guide, logistics, planner = create_agents(llm, weather_summary)
        
        # Create tasks with weather context
        logistics_task = Task(
            description=f"""Research comprehensive logistics for a trip to {request.destination} from {request.start_date} to {request.end_date} with a {request.budget} budget.

Include:
- Accommodation recommendations (hotels, hostels, Airbnb) suitable for {request.budget} budget
- Transportation options (flights, trains, local transport)
- Visa requirements and documentation needed
- Safety information and local customs
- Weather-appropriate packing suggestions

Weather Context: {weather_summary}

Provide practical, actionable information in a well-structured format.""",
            agent=logistics,
            expected_output="Detailed logistics report covering accommodation, transport, documentation, and weather considerations"
        )

        guide_task = Task(
            description=f"""Find the best {request.interests} activities, attractions, and experiences in {request.destination} for a {days}-day trip.

Consider:
- Top attractions related to {request.interests}
- Hidden gems and local favorites
- Indoor and outdoor activity options
- Cultural experiences and local cuisine
- Activities suitable for current weather conditions

Weather Context: {weather_summary}

Focus on creating memorable experiences that match the traveler's interests.""",
            agent=guide,
            expected_output="Comprehensive list of recommended activities and attractions with weather suitability notes"
        )

        plan_task = Task(
            description=f"""Create a detailed {days}-day itinerary for {request.destination} from {request.start_date} to {request.end_date}.

Requirements:
- Daily schedule with specific activities and timing
- Balance of {request.interests} with practical logistics
- Weather-appropriate activity scheduling (indoor for bad weather, outdoor for good weather)
- Realistic travel times between locations
- Meal suggestions and rest periods
- Backup indoor activities for unexpected weather

Use the logistics and attractions information from other team members.

Weather Context: {weather_summary}""",
            agent=planner,
            context=[logistics_task, guide_task],
            expected_output="Complete day-by-day itinerary with times, activities, and weather contingencies"
        )

        # Execute crew
        crew = Crew(
            agents=[guide, logistics, planner],
            tasks=[logistics_task, guide_task, plan_task],
            process=Process.sequential,
            verbose=True
        )
        
        logger.info("Starting crew execution...")
        result = crew.kickoff()
        logger.info("Crew execution completed")

        # Parse results - CrewAI returns different formats
        def extract_task_result(task_result):
            if hasattr(task_result, 'raw'):
                return {"content": task_result.raw}
            elif hasattr(task_result, 'result'):
                return {"content": str(task_result.result)}
            elif isinstance(task_result, str):
                return {"content": task_result}
            else:
                return {"content": str(task_result)}
        
        # Handle the crew result
        if hasattr(result, 'tasks_output'):
            tasks_results = result.tasks_output
        elif isinstance(result, list):
            tasks_results = result
        else:
            # Single result - extract from final task
            tasks_results = [result] * 3  # Duplicate for all three outputs
        
        # Ensure we have results for all tasks
        logistics_result = extract_task_result(tasks_results[0]) if len(tasks_results) > 0 else {"content": "Logistics information not available"}
        guide_result = extract_task_result(tasks_results[1]) if len(tasks_results) > 1 else {"content": "Guide information not available"}
        itinerary_result = extract_task_result(tasks_results[2]) if len(tasks_results) > 2 else {"content": "Itinerary not available"}
        
        return TravelResponse(
            itinerary=itinerary_result,
            guide=guide_result,
            logistics=logistics_result,
            weather=weather_data,
            language="en"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Planning error: {str(e)}")
        raise HTTPException(500, detail=f"Failed to generate travel plan: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
        
        # Test Ollama connection by checking available models
        response = requests.get(f"{ollama_url}/api/tags", timeout=10)
        
        if response.status_code == 200:
            models_data = response.json()
            available_models = [model["name"] for model in models_data.get("models", [])]
            
            # Check if our target model is available
            model_available = any(model_name in model for model in available_models)
            
            if model_available:
                # Test actual LLM call
                try:
                    llm = get_llm()
                    test_response = llm._call("Say 'Hello' in one word")
                    
                    return {
                        "status": "healthy",
                        "ollama": "connected",
                        "ollama_url": ollama_url,
                        "model": model_name,
                        "model_available": True,
                        "available_models": available_models,
                        "test_response": test_response.strip()[:50]
                    }
                except Exception as llm_error:
                    return {
                        "status": "partially_healthy",
                        "ollama": "connected",
                        "ollama_url": ollama_url,
                        "model": model_name,
                        "model_available": True,
                        "available_models": available_models,
                        "llm_test_error": str(llm_error)
                    }
            else:
                return {
                    "status": "unhealthy",
                    "ollama": "connected",
                    "ollama_url": ollama_url,
                    "model": model_name,
                    "model_available": False,
                    "available_models": available_models,
                    "error": f"Model '{model_name}' not found. Available models: {available_models}"
                }
        else:
            return {
                "status": "unhealthy",
                "ollama": "disconnected",
                "ollama_url": ollama_url,
                "error": f"Cannot connect to Ollama at {ollama_url}. Status: {response.status_code}"
            }
            
    except requests.exceptions.ConnectionError:
        return {
            "status": "unhealthy",
            "ollama": "disconnected", 
            "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
            "error": "Connection refused. Make sure Ollama is running with 'ollama serve'"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
            "error": str(e)
        }

# Get available models
@app.get("/models")
async def get_models():
    try:
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        response = requests.get(f"{ollama_url}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json()
            return {"models": [model["name"] for model in models.get("models", [])]}
        else:
            return {"error": "Cannot fetch models from Ollama"}
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
    print("Starting Travel Planner with Ollama integration...")
    print("Make sure Ollama is running with: ollama serve")
    print("And that you have llama3.2 model: ollama pull llama3.2")
    uvicorn.run(
        "travel:app",  # Use import string format
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )