import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Загружаем переменные окружения (включая API-ключ)
load_dotenv()

# Создаем приложение FastAPI
app = FastAPI()

# Создаем клиента OpenAI.
# Ключ будет автоматически подхвачен из переменных окружения Vercel.
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Модель для входящих данных (ожидаем получить username)
class AnalyzeRequest(BaseModel):
    username: str

@app.post("/analyze")
def analyze_instagram_profile(request: AnalyzeRequest):
    try:
        # Формируем промпт для GPT
        prompt_text = f"""
        Проанализируй следующий профиль в Instagram по его никнейму: '{request.username}'.

        Твоя задача — создать детальное саммари по следующим пунктам:
        1.  **Имя и Категория:** Попробуй определить реальное имя и категорию блога (например, "Личный блог", "Путешествия", "Мода").
        2.  **Внешние ссылки:** Если есть ссылки в профиле, укажи, куда они ведут и что это говорит о деятельности человека.
        3.  **Персонализированный анализ (Стиль жизни и Визуальный код):**
            *   **География и тип путешествий:** Определи по визуальному ряду, в каких локациях бывает человек (например, Азия, Европа, США) и какой тип отдыха это (пляжный, городской, люкс, приключения).
            *   **Маркеры статуса и благосостояния:** Есть ли в кадре элементы, указывающие на уровень жизни (дорогие часы, автомобили, брендовая одежда, частные самолеты, рестораны).
            *   **Мода и стиль (Fashion Analysis):** Опиши стиль одежды. Это "тихая роскошь" (quiet luxury), стритвир, кэжуал? Упоминаются ли или видны ли люксовые бренды?
        4.  **Сводка и гипотеза:** Сделай краткое заключение. Кто этот человек, чем он, вероятно, занимается и для кого ведет свой блог?

        Представь результат в виде структурированного текста на русском языке.
        """

        # Отправляем запрос в OpenAI
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],
            model="gpt-4-turbo", # или другая доступная модель
        )

        # Получаем и возвращаем результат
        analysis_result = chat_completion.choices[0].message.content
        return {"analysis": analysis_result}

    except Exception as e:
        # Если что-то пошло не так, возвращаем ошибку
        print(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
