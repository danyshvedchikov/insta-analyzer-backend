from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import requests
import os
import base64
from typing import Optional

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
    photo_url: Optional[str] = None
    image_base64: Optional[str] = None


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


def get_image_bytes(photo_url: Optional[str], image_base64: Optional[str]) -> bytes:
    """Get image bytes from URL or base64 string"""
    if image_base64:
        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        return base64.b64decode(image_base64)
    elif photo_url:
        response = requests.get(photo_url, timeout=30)
        response.raise_for_status()
        return response.content
    else:
        raise ValueError("Either photo_url or image_base64 must be provided")


def analyze_with_gpt4_vision(photo_url: Optional[str], image_base64: Optional[str]) -> dict:
    """Analyze photo using GPT-4 Vision for filter and manipulation detection"""
    result = {
        "has_filters": None,
        "filter_type": None,
        "manipulation_signs": None,
        "authenticity_score": None,
        "analysis": None,
        "error": None
    }
    
    try:
        # Prepare image for GPT-4 Vision
        if image_base64:
            # Ensure proper base64 format
            if ',' in image_base64:
                image_data = image_base64
            else:
                image_data = f"data:image/jpeg;base64,{image_base64}"
        else:
            image_data = photo_url
        
        # Technical analysis prompt - focuses on image processing artifacts
        prompt = """You are a technical image analysis expert. Analyze this photo for digital modifications.

Analyze and report:

1. DIGITAL OVERLAYS: Are there any AR filters, face masks, stickers, or digital overlays added to the image? (Snapchat, Instagram, TikTok filters etc.)

2. BEAUTY PROCESSING: Is there evidence of skin smoothing, face reshaping, eye enlargement, or other beauty filter effects?

3. IMAGE MANIPULATION: Are there signs of photo editing, retouching, or AI generation artifacts?

4. AUTHENTICITY ESTIMATE: Rate from 0-100% how natural/unedited this photo appears (100% = completely natural)

Respond in this exact format:
FILTERS: [Yes/No] - [filter type if present]
BEAUTY: [Yes/No] - [description]
EDITING: [Yes/No] - [description]
SCORE: [number]%
SUMMARY: [one sentence conclusion in Russian]"""

        response = client.chat.completions.create(
            model="gpt-4o",  # Using gpt-4o for better vision capabilities
            messages=[
                {
                    "role": "system",
                    "content": "You are a technical image forensics expert. Your task is to analyze photos for digital modifications, filters, and editing. Always provide objective technical analysis."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data, "detail": "high"}
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        analysis_text = response.choices[0].message.content
        result["analysis"] = analysis_text
        
        # Parse the response
        lines = analysis_text.upper()
        
        # Check for filters
        if "FILTERS: YES" in lines or "FILTERS:YES" in lines:
            result["has_filters"] = True
        elif "FILTERS: NO" in lines or "FILTERS:NO" in lines:
            result["has_filters"] = False
        
        # Try to extract authenticity score
        import re
        match = re.search(r'SCORE[:\s]*(\d+)', lines)
        if match:
            result["authenticity_score"] = int(match.group(1))
            
    except Exception as e:
        error_msg = str(e)
        # Check if it's a content policy error
        if "content" in error_msg.lower() or "policy" in error_msg.lower() or "safety" in error_msg.lower():
            result["error"] = "Изображение не может быть проанализировано (ограничения модели)"
        else:
            result["error"] = f"GPT Vision: {error_msg[:100]}"
    
    return result


def analyze_with_sightengine(photo_url: Optional[str], image_base64: Optional[str]) -> dict:
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
        "raw_faces": None  # For debugging
    }
    
    try:
        if image_base64:
            # Upload raw binary image
            image_bytes = get_image_bytes(None, image_base64)
            
            files = {
                'media': ('image.jpg', image_bytes, 'image/jpeg')
            }
            data = {
                'models': 'quality,genai,face-attributes',
                'api_user': SIGHTENGINE_USER,
                'api_secret': SIGHTENGINE_SECRET
            }
            
            response = requests.post(
                'https://api.sightengine.com/1.0/check.json',
                files=files,
                data=data,
                timeout=30
            )
        else:
            # Use URL
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
        
        if data.get("status") == "success":
            # Parse quality score (0-1 scale)
            if "quality" in data and "score" in data["quality"]:
                result["quality_score"] = round(data["quality"]["score"] * 100, 1)
            
            # Parse AI-generated score (0-1 scale, in "type" object)
            if "type" in data and "ai_generated" in data["type"]:
                result["ai_generated"] = round(data["type"]["ai_generated"] * 100, 1)
            
            # Parse face attributes - FIXED PARSING
            if "faces" in data and len(data["faces"]) > 0:
                result["face_detected"] = True
                face = data["faces"][0]
                result["raw_faces"] = face  # Store raw data for debugging
                
                # Face attributes can be nested in different ways
                # Try direct access first
                if "attributes" in face:
                    attrs = face["attributes"]
                    
                    # Quality can be a string or nested object
                    if "quality" in attrs:
                        q = attrs["quality"]
                        if isinstance(q, str):
                            result["face_quality"] = q
                        elif isinstance(q, dict) and "value" in q:
                            result["face_quality"] = q["value"]
                    
                    # Obstruction
                    if "obstruction" in attrs:
                        o = attrs["obstruction"]
                        if isinstance(o, str):
                            result["face_obstruction"] = o
                        elif isinstance(o, dict) and "value" in o:
                            result["face_obstruction"] = o["value"]
                    
                    # Angle
                    if "angle" in attrs:
                        a = attrs["angle"]
                        if isinstance(a, str):
                            result["face_angle"] = a
                        elif isinstance(a, dict) and "value" in a:
                            result["face_angle"] = a["value"]
                    
                    # Filters - IMPORTANT
                    if "filters" in attrs:
                        f = attrs["filters"]
                        if isinstance(f, bool):
                            result["face_filters"] = f
                        elif isinstance(f, dict) and "value" in f:
                            result["face_filters"] = f["value"]
                        elif f == "true" or f == True:
                            result["face_filters"] = True
                        elif f == "false" or f == False:
                            result["face_filters"] = False
                
                # Check for sunglasses at face level
                if "sunglasses" in face:
                    s = face["sunglasses"]
                    if isinstance(s, bool):
                        result["sunglasses"] = s
                    elif isinstance(s, dict) and "value" in s:
                        result["sunglasses"] = s["value"]
        else:
            result["error"] = data.get("error", {}).get("message", "Unknown error")
            
    except requests.exceptions.Timeout:
        result["error"] = "Sightengine API timeout"
    except Exception as e:
        result["error"] = f"Sightengine error: {str(e)}"
    
    return result


def analyze_with_hive(photo_url: Optional[str], image_base64: Optional[str]) -> dict:
    """Analyze photo using Hive AI API for AI-generated detection"""
    result = {
        "ai_generated": None,
        "ai_source": None,
        "deepfake": None,
        "error": None
    }
    
    if not HIVE_API_KEY:
        result["error"] = "Hive API key not configured"
        return result
    
    try:
        headers = {
            "Authorization": f"Token {HIVE_API_KEY}"
        }
        
        if image_base64:
            # Upload as multipart form data
            image_bytes = get_image_bytes(None, image_base64)
            
            files = {
                'media': ('image.jpg', image_bytes, 'image/jpeg')
            }
            
            response = requests.post(
                "https://api.thehive.ai/api/v2/task/sync",
                headers=headers,
                files=files,
                timeout=30
            )
        else:
            # Use URL - IMPORTANT: use data= not json=
            payload = {"url": photo_url}
            
            response = requests.post(
                "https://api.thehive.ai/api/v2/task/sync",
                headers=headers,
                data=payload,
                timeout=30
            )
        
        data = response.json()
        
        # Parse Hive AI response
        if "status" in data and isinstance(data["status"], list) and len(data["status"]) > 0:
            status_item = data["status"][0]
            
            # Check for error in status
            if "status" in status_item:
                inner_status = status_item["status"]
                if inner_status.get("code") != "0":
                    result["error"] = inner_status.get("message", "Unknown Hive error")
                    return result
            
            if "response" in status_item and "output" in status_item["response"]:
                output = status_item["response"]["output"]
                
                if len(output) > 0 and "classes" in output[0]:
                    classes = output[0]["classes"]
                    
                    ai_generated_score = None
                    best_source = None
                    best_source_score = 0
                    deepfake_score = None
                    
                    ai_generators = [
                        "sora", "pika", "haiper", "kling", "luma", "hedra", "runway",
                        "hailuo", "mochi", "flux", "hallo", "hunyuan", "recraft",
                        "leonardo", "dalle", "midjourney", "stablediffusion",
                        "stablediffusionxl", "adobefirefly", "gan", "ideogram",
                        "other_image_generators"
                    ]
                    
                    for cls in classes:
                        class_name = cls.get("class", "")
                        score = cls.get("score", 0)
                        
                        if class_name == "ai_generated":
                            ai_generated_score = score
                        
                        if class_name == "deepfake":
                            deepfake_score = score
                        
                        if class_name in ai_generators and score > best_source_score:
                            best_source = class_name
                            best_source_score = score
                    
                    if ai_generated_score is not None:
                        result["ai_generated"] = round(ai_generated_score * 100, 1)
                    
                    if deepfake_score is not None:
                        result["deepfake"] = round(deepfake_score * 100, 1)
                    
                    if best_source and best_source_score > 0.1:
                        result["ai_source"] = f"{best_source} ({round(best_source_score * 100, 1)}%)"
        else:
            result["error"] = "Unexpected Hive API response format"
                                
    except requests.exceptions.Timeout:
        result["error"] = "Hive API timeout"
    except Exception as e:
        result["error"] = f"Hive error: {str(e)}"
    
    return result


def format_analysis_result(sightengine_result: dict, hive_result: dict, gpt_result: dict) -> str:
    """Format the analysis results in Russian with emoji"""
    lines = []
    
    lines.append("📸 **АНАЛИЗ ФОТОГРАФИИ**")
    lines.append("")
    
    # Filter Detection Section (MOST IMPORTANT)
    lines.append("🎭 **Фильтры и маски:**")
    
    # GPT-4 Vision filter detection
    if gpt_result.get("has_filters") is True:
        lines.append("  • GPT-4 Vision: ✅ Обнаружены фильтры")
    elif gpt_result.get("has_filters") is False:
        lines.append("  • GPT-4 Vision: ❌ Фильтры не обнаружены")
    elif gpt_result.get("error"):
        lines.append(f"  • GPT-4 Vision: ⚠️ {gpt_result['error']}")
    else:
        lines.append("  • GPT-4 Vision: Анализ недоступен")
    
    # Sightengine filter detection
    if sightengine_result.get("face_filters") is True:
        lines.append("  • Sightengine: ✅ Обнаружены фильтры на лице")
    elif sightengine_result.get("face_filters") is False:
        lines.append("  • Sightengine: ❌ Фильтры не обнаружены")
    elif sightengine_result.get("face_detected"):
        lines.append("  • Sightengine: Данные о фильтрах недоступны")
    
    lines.append("")
    
    # AI Generation Detection Section
    lines.append("🤖 **ИИ-генерация (полностью созданные ИИ):**")
    
    # Sightengine AI detection
    if sightengine_result.get("ai_generated") is not None:
        ai_score = sightengine_result["ai_generated"]
        if ai_score < 20:
            verdict = "✅ Реальное фото"
        elif ai_score < 50:
            verdict = "⚠️ Возможна обработка"
        else:
            verdict = "🚨 Вероятно ИИ"
        lines.append(f"  • Sightengine: {ai_score}% {verdict}")
    else:
        lines.append("  • Sightengine: Данные недоступны")
    
    # Hive AI detection
    if hive_result.get("ai_generated") is not None:
        ai_score = hive_result["ai_generated"]
        if ai_score < 20:
            verdict = "✅ Реальное фото"
        elif ai_score < 50:
            verdict = "⚠️ Возможна обработка"
        else:
            verdict = "🚨 Вероятно ИИ"
        lines.append(f"  • Hive AI: {ai_score}% {verdict}")
        
        if hive_result.get("ai_source"):
            lines.append(f"    Источник: {hive_result['ai_source']}")
    else:
        if hive_result.get("error"):
            lines.append(f"  • Hive AI: ⚠️ {hive_result['error'][:50]}")
        else:
            lines.append("  • Hive AI: Данные недоступны")
    
    # Deepfake detection
    if hive_result.get("deepfake") is not None:
        df_score = hive_result["deepfake"]
        if df_score < 20:
            verdict = "✅ Не дипфейк"
        elif df_score < 50:
            verdict = "⚠️ Возможен дипфейк"
        else:
            verdict = "🚨 Вероятно дипфейк"
        lines.append(f"  • Дипфейк: {df_score}% {verdict}")
    
    lines.append("")
    
    # Quality Section
    lines.append("📊 **Качество:**")
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
        lines.append(f"  • Качество: {quality}/100 ({quality_text})")
    
    lines.append("")
    
    # Face Analysis Section
    lines.append("👤 **Анализ лица:**")
    if sightengine_result.get("face_detected"):
        lines.append("  • Лицо: ✅ Обнаружено")
        
        face_quality = sightengine_result.get("face_quality")
        if face_quality and face_quality != "unknown":
            quality_map = {"perfect": "Идеальное", "high": "Высокое", "medium": "Среднее", "low": "Низкое"}
            lines.append(f"  • Качество лица: {quality_map.get(face_quality, face_quality)}")
        
        face_obstruction = sightengine_result.get("face_obstruction")
        if face_obstruction and face_obstruction != "unknown":
            obstruction_map = {"none": "Нет", "light": "Легкое", "medium": "Среднее", "heavy": "Сильное", "extreme": "Экстремальное", "complete": "Полное"}
            lines.append(f"  • Препятствия: {obstruction_map.get(face_obstruction, face_obstruction)}")
        
        sunglasses = sightengine_result.get("sunglasses")
        if sunglasses is not None:
            lines.append(f"  • Очки: {'Да' if sunglasses else 'Нет'}")
    else:
        lines.append("  • Лицо: ❌ Не обнаружено")
    
    lines.append("")
    
    # GPT-4 Vision detailed analysis
    if gpt_result.get("analysis") and not gpt_result.get("error"):
        lines.append("🔍 **Детальный анализ (GPT-4 Vision):**")
        for line in gpt_result["analysis"].split('\n'):
            if line.strip():
                lines.append(f"  {line}")
        lines.append("")
    
    # Overall verdict
    lines.append("📋 **ИТОГ:**")
    
    # Calculate overall authenticity
    authenticity_scores = []
    
    # From GPT-4 Vision
    if gpt_result.get("authenticity_score") is not None:
        authenticity_scores.append(gpt_result["authenticity_score"])
    
    # From AI detection (inverse)
    ai_scores = []
    if sightengine_result.get("ai_generated") is not None:
        ai_scores.append(sightengine_result["ai_generated"])
    if hive_result.get("ai_generated") is not None:
        ai_scores.append(hive_result["ai_generated"])
    
    if ai_scores:
        avg_ai = sum(ai_scores) / len(ai_scores)
        authenticity_scores.append(100 - avg_ai)
    
    # Determine filter penalty
    filter_penalty = 0
    if gpt_result.get("has_filters") is True:
        filter_penalty = 30
    elif sightengine_result.get("face_filters") is True:
        filter_penalty = 25
    
    if authenticity_scores:
        avg_authenticity = sum(authenticity_scores) / len(authenticity_scores)
        final_score = max(0, avg_authenticity - filter_penalty)
        
        if final_score >= 80:
            verdict = "✅ Фото выглядит подлинным"
        elif final_score >= 50:
            verdict = "⚠️ Фото обработано или с фильтрами"
        else:
            verdict = "🚨 Сильная обработка или ИИ-генерация"
        
        lines.append(f"  • Подлинность: {round(final_score, 1)}%")
        lines.append(f"  • Вердикт: {verdict}")
        
        if filter_penalty > 0:
            lines.append(f"  • Примечание: Обнаружены фильтры (-{filter_penalty}%)")
    else:
        lines.append("  • Недостаточно данных для оценки")
    
    return "\n".join(lines)


@app.post("/analyze-photo")
async def analyze_photo(request: PhotoRequest):
    """Analyze photo for AI-generation, filters, quality, and face attributes"""
    try:
        # Validate input
        if not request.photo_url and not request.image_base64:
            raise HTTPException(
                status_code=400, 
                detail="Either photo_url or image_base64 must be provided"
            )
        
        # Run all analyses
        sightengine_result = analyze_with_sightengine(request.photo_url, request.image_base64)
        hive_result = analyze_with_hive(request.photo_url, request.image_base64)
        gpt_result = analyze_with_gpt4_vision(request.photo_url, request.image_base64)
        
        # Format the results
        formatted_result = format_analysis_result(sightengine_result, hive_result, gpt_result)
        
        return {
            "analysis": formatted_result,
            "debug": {
                "sightengine": {k: v for k, v in sightengine_result.items() if k != "raw_faces"},
                "hive": hive_result,
                "gpt_vision": {k: v for k, v in gpt_result.items() if k != "analysis"}
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error analyzing photo: {str(e)}"
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
