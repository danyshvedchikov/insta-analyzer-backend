from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import requests
import os
import json
import traceback

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Sightengine credentials
SIGHTENGINE_USER = os.getenv("SIGHTENGINE_USER")
SIGHTENGINE_SECRET = os.getenv("SIGHTENGINE_SECRET")

# Hive AI credentials
HIVE_API_KEY = os.getenv("HIVE_API_KEY")


class ProfileRequest(BaseModel):
    username: str
    bio: str
    posts_count: int
    followers_count: int
    following_count: int


class PhotoRequest(BaseModel):
    photo_url: str


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Instagram Analyzer API is running"}


@app.post("/analyze")
async def analyze_profile(request: ProfileRequest):
    """Analyze Instagram profile using GPT-4o-mini"""
    try:
        prompt = f"""Проанализируй Instagram профиль и дай детальную оценку:

Имя пользователя: {request.username}
Биография: {request.bio}
Количество постов: {request.posts_count}
Подписчики: {request.followers_count}
Подписки: {request.following_count}

Дай анализ по следующим критериям:
1. 🎯 Оценка подлинности профиля (0-100%)
2. 📊 Анализ активности и вовлеченности
3. 🔍 Признаки бота или фейкового аккаунта
4. 💡 Рекомендации

Формат ответа должен быть структурированным с emoji и разделами."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты эксперт по анализу социальных сетей. Отвечай на русском языке."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )

        return {"analysis": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def analyze_with_sightengine(photo_url: str) -> dict:
    """Analyze photo using Sightengine API"""
    result = {
        "ai_generated": None,
        "quality_score": None,
        "face_detected": False,
        "face_quality": None,
        "face_obstruction": None,
        "face_angle": None,
        "face_filters": None,
        "sunglasses": None,
        "error": None,
        "raw_response": None
    }
    
    try:
        # Use multiple models: quality, genai, face-attributes
        params = {
            'url': photo_url,
            'models': 'quality,genai,face-attributes',
            'api_user': SIGHTENGINE_USER,
            'api_secret': SIGHTENGINE_SECRET
        }
        
        response = requests.get(
            'https://api.sightengine.com/1.0/check.json',
            params=params,
            timeout=30
        )
        
        data = response.json()
        result["raw_response"] = data
        
        if data.get("status") == "success":
            # Parse quality score (0-1 scale)
            if "quality" in data and "score" in data["quality"]:
                result["quality_score"] = round(data["quality"]["score"] * 100, 1)
            
            # Parse AI-generated score (0-1 scale, in "type" object)
            if "type" in data and "ai_generated" in data["type"]:
                result["ai_generated"] = round(data["type"]["ai_generated"] * 100, 1)
            
            # Parse face attributes
            if "faces" in data and len(data["faces"]) > 0:
                result["face_detected"] = True
                face = data["faces"][0]
                
                # Get face attributes
                if "attributes" in face:
                    attrs = face["attributes"]
                    result["face_quality"] = attrs.get("quality", "unknown")
                    result["face_obstruction"] = attrs.get("obstruction", "unknown")
                    result["face_angle"] = attrs.get("angle", "unknown")
                    result["face_filters"] = attrs.get("filters", None)
                
                # Check for sunglasses
                if "sunglasses" in face:
                    result["sunglasses"] = face["sunglasses"]
        else:
            result["error"] = data.get("error", {}).get("message", "Unknown error")
            
    except requests.exceptions.Timeout:
        result["error"] = "Sightengine API timeout"
    except Exception as e:
        result["error"] = f"Sightengine error: {str(e)}"
    
    return result


def analyze_with_hive(photo_url: str) -> dict:
    """Analyze photo using Hive AI API for AI-generated detection"""
    result = {
        "ai_generated": None,
        "ai_source": None,
        "error": None,
        "raw_response": None
    }
    
    try:
        headers = {
            "Authorization": f"Token {HIVE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "url": photo_url
        }
        
        response = requests.post(
            "https://api.thehive.ai/api/v2/task/sync",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        data = response.json()
        result["raw_response"] = data
        
        # Parse Hive AI response
        # Response structure: status[0].response.output[0].classes[]
        if "status" in data and len(data["status"]) > 0:
            status_item = data["status"][0]
            
            if "response" in status_item and "output" in status_item["response"]:
                output = status_item["response"]["output"]
                
                if len(output) > 0 and "classes" in output[0]:
                    classes = output[0]["classes"]
                    
                    ai_generated_score = None
                    best_source = None
                    best_source_score = 0
                    
                    # List of known AI generators
                    ai_generators = [
                        "sora", "pika", "haiper", "kling", "luma", "hedra", "runway",
                        "hailuo", "mochi", "flux", "hallo", "hunyuan", "recraft",
                        "leonardo", "luminagpt", "var", "liveportrait", "mcnet",
                        "pyramidflows", "sadtalker", "aniportrait", "cogvideos",
                        "makeittalk", "sdxlinpaint", "stablediffusioninpaint",
                        "bingimagecreator", "adobefirefly", "lcm", "dalle", "pixart",
                        "glide", "stablediffusion", "imagen", "amused", "stablecascade",
                        "midjourney", "deepfloyd", "gan", "stablediffusionxl",
                        "vqdiffusion", "kandinsky", "wuerstchen", "titan", "ideogram",
                        "sana", "emu3", "omnigen", "flashvideo", "transpixar", "cosmos",
                        "janus", "dmd2", "switti", "4o", "grok", "wan", "infinity",
                        "veo3", "imagen4", "other_image_generators"
                    ]
                    
                    for cls in classes:
                        class_name = cls.get("class", "")
                        score = cls.get("score", 0)
                        
                        # Get AI-generated score
                        if class_name == "ai_generated":
                            ai_generated_score = score
                        
                        # Find the best matching AI source
                        if class_name in ai_generators and score > best_source_score:
                            best_source = class_name
                            best_source_score = score
                    
                    if ai_generated_score is not None:
                        result["ai_generated"] = round(ai_generated_score * 100, 1)
                    
                    if best_source and best_source_score > 0.1:
                        result["ai_source"] = f"{best_source} ({round(best_source_score * 100, 1)}%)"
        
        # Alternative response structure (direct status without array)
        elif "status" in data and isinstance(data["status"], dict):
            if data["status"].get("code") == "0" or data["status"].get("message") == "SUCCESS":
                if "response" in data and "output" in data["response"]:
                    output = data["response"]["output"]
                    if len(output) > 0 and "classes" in output[0]:
                        classes = output[0]["classes"]
                        for cls in classes:
                            if cls.get("class") == "ai_generated":
                                result["ai_generated"] = round(cls.get("score", 0) * 100, 1)
                                break
                                
    except requests.exceptions.Timeout:
        result["error"] = "Hive API timeout"
    except Exception as e:
        result["error"] = f"Hive error: {str(e)}\n{traceback.format_exc()}"
    
    return result


def format_analysis_result(sightengine_result: dict, hive_result: dict) -> str:
    """Format the analysis results in Russian with emoji"""
    lines = []
    
    lines.append("📸 **АНАЛИЗ ФОТОГРАФИИ**")
    lines.append("")
    
    # AI Generation Detection Section
    lines.append("🤖 **Обнаружение ИИ-генерации:**")
    
    # Sightengine AI detection
    if sightengine_result.get("ai_generated") is not None:
        ai_score = sightengine_result["ai_generated"]
        if ai_score < 20:
            verdict = "✅ Вероятно реальное фото"
        elif ai_score < 50:
            verdict = "⚠️ Возможно отредактировано"
        else:
            verdict = "🚨 Вероятно ИИ-генерация"
        lines.append(f"  • Sightengine: {ai_score}% {verdict}")
    else:
        lines.append(f"  • Sightengine: Данные недоступны")
    
    # Hive AI detection
    if hive_result.get("ai_generated") is not None:
        ai_score = hive_result["ai_generated"]
        if ai_score < 20:
            verdict = "✅ Вероятно реальное фото"
        elif ai_score < 50:
            verdict = "⚠️ Возможно отредактировано"
        else:
            verdict = "🚨 Вероятно ИИ-генерация"
        lines.append(f"  • Hive AI: {ai_score}% {verdict}")
        
        if hive_result.get("ai_source"):
            lines.append(f"  • Возможный источник: {hive_result['ai_source']}")
    else:
        if hive_result.get("error"):
            lines.append(f"  • Hive AI: Ошибка - {hive_result['error'][:100]}")
        else:
            lines.append(f"  • Hive AI: Данные недоступны")
    
    lines.append("")
    
    # Quality Section
    lines.append("📊 **Качество изображения:**")
    if sightengine_result.get("quality_score") is not None:
        quality = sightengine_result["quality_score"]
        if quality >= 85:
            quality_text = "Отличное"
        elif quality >= 60:
            quality_text = "Хорошее"
        elif quality >= 45:
            quality_text = "Среднее"
        else:
            quality_text = "Низкое"
        lines.append(f"  • Оценка качества: {quality}/100 ({quality_text})")
    else:
        lines.append(f"  • Оценка качества: Данные недоступны")
    
    lines.append("")
    
    # Face Analysis Section
    lines.append("👤 **Анализ лица:**")
    if sightengine_result.get("face_detected"):
        lines.append(f"  • Лицо обнаружено: ✅ Да")
        
        # Face quality
        face_quality = sightengine_result.get("face_quality")
        if face_quality:
            quality_map = {
                "perfect": "Идеальное",
                "high": "Высокое",
                "medium": "Среднее",
                "low": "Низкое"
            }
            lines.append(f"  • Качество лица: {quality_map.get(face_quality, face_quality)}")
        
        # Face obstruction
        face_obstruction = sightengine_result.get("face_obstruction")
        if face_obstruction:
            obstruction_map = {
                "none": "Нет препятствий",
                "light": "Легкое",
                "medium": "Среднее",
                "heavy": "Сильное",
                "extreme": "Экстремальное",
                "complete": "Полное"
            }
            lines.append(f"  • Препятствия: {obstruction_map.get(face_obstruction, face_obstruction)}")
        
        # Face angle
        face_angle = sightengine_result.get("face_angle")
        if face_angle:
            angle_map = {
                "straight": "Прямой",
                "side": "Боковой",
                "back": "Сзади"
            }
            lines.append(f"  • Угол лица: {angle_map.get(face_angle, face_angle)}")
        
        # Filters
        face_filters = sightengine_result.get("face_filters")
        if face_filters is not None:
            lines.append(f"  • Фильтры на лице: {'Да' if face_filters else 'Нет'}")
        
        # Sunglasses
        sunglasses = sightengine_result.get("sunglasses")
        if sunglasses is not None:
            lines.append(f"  • Солнечные очки: {'Да' if sunglasses else 'Нет'}")
    else:
        lines.append(f"  • Лицо обнаружено: ❌ Нет")
    
    lines.append("")
    
    # Overall verdict
    lines.append("📋 **Итоговая оценка:**")
    
    # Calculate overall authenticity
    ai_scores = []
    if sightengine_result.get("ai_generated") is not None:
        ai_scores.append(sightengine_result["ai_generated"])
    if hive_result.get("ai_generated") is not None:
        ai_scores.append(hive_result["ai_generated"])
    
    if ai_scores:
        avg_ai = sum(ai_scores) / len(ai_scores)
        authenticity = 100 - avg_ai
        
        if authenticity >= 80:
            verdict = "✅ Фото выглядит подлинным"
        elif authenticity >= 50:
            verdict = "⚠️ Фото может быть отредактировано"
        else:
            verdict = "🚨 Высокая вероятность ИИ-генерации"
        
        lines.append(f"  • Подлинность: {round(authenticity, 1)}%")
        lines.append(f"  • Вердикт: {verdict}")
    else:
        lines.append(f"  • Недостаточно данных для оценки")
    
    # Add errors if any
    errors = []
    if sightengine_result.get("error"):
        errors.append(f"Sightengine: {sightengine_result['error']}")
    if hive_result.get("error"):
        errors.append(f"Hive: {hive_result['error']}")
    
    if errors:
        lines.append("")
        lines.append("⚠️ **Ошибки:**")
        for error in errors:
            lines.append(f"  • {error[:150]}")
    
    return "\n".join(lines)


@app.post("/analyze-photo")
async def analyze_photo(request: PhotoRequest):
    """Analyze photo for AI-generation, quality, and face attributes"""
    try:
        # Run both analyses
        sightengine_result = analyze_with_sightengine(request.photo_url)
        hive_result = analyze_with_hive(request.photo_url)
        
        # Format the results
        formatted_result = format_analysis_result(sightengine_result, hive_result)
        
        return {
            "analysis": formatted_result,
            "debug": {
                "sightengine": sightengine_result,
                "hive": hive_result
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error analyzing photo: {str(e)}\n{traceback.format_exc()}"
        )


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "sightengine_configured": bool(SIGHTENGINE_USER and SIGHTENGINE_SECRET),
        "hive_configured": bool(HIVE_API_KEY),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
    }
