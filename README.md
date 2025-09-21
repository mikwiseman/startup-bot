# Telegram Startup Advisor Bot

Телеграм-бот, который собирает информацию о стартап-идее и запрашивает обратную связь у модели OpenAI. Результат содержит краткую сводку, оценку по ключевым критериям и список рекомендаций.

## Требования

- Python 3.10+
- Токен Telegram-бота
- API-ключ OpenAI с доступом к моделям семейства GPT-4o или совместимым

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Создайте файл `.env` (можно использовать `.env.example` как шаблон):

```
TELEGRAM_BOT_TOKEN=ваш-токен
OPENAI_API_KEY=ваш-ключ
OPENAI_MODEL=gpt-4o-mini  # при необходимости замените на другую модель
```

## Запуск

```bash
python3 bot.py
```

Бот начнет диалог по команде `/start` и задаст серию вопросов об идее. После получения всех ответов он отправит оценку и рекомендации. Командой `/cancel` можно прервать текущий опрос, `/help` — получить подсказку.

## Публикация на GitHub

1. Убедитесь, что каталог чистый:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```
2. Добавьте удаленный репозиторий и запушьте код:
   ```bash
   git remote add origin https://github.com/mikwiseman/startup-bot.git
   git push -u origin main  # используйте master, если выбранная ветка называется иначе
   ```

## Деплой на Railway

Файлы `Procfile` и `railway.json` уже подготовлены: Railway автоматически запустит команду `python bot.py`.

1. Установите [Railway CLI](https://docs.railway.app/develop/cli) и авторизуйтесь (`railway login`).
2. Находясь в корне проекта, выполните инициализацию:
   ```bash
   railway init
   railway up
   ```
3. Задайте переменные окружения (можно через Web UI или CLI):
   ```bash
   railway variables set TELEGRAM_BOT_TOKEN=... OPENAI_API_KEY=... OPENAI_MODEL=gpt-4o-mini
   ```
4. После деплоя сервис запустится как фоновой воркер. Логи можно просматривать командой `railway logs`.

## Настройка и доработка

- Список вопросов можно изменить в массиве `QUESTION_FLOW` внутри `bot.py`.
- Для изменения модели OpenAI обновите переменную `OPENAI_MODEL` в окружении.
- При желании можно добавить хранение истории в базе данных или интеграции с другими сервисами, используя данные из `context.user_data`.
