import asyncio
import json
import logging
import os
import sqlite3
from threading import Lock
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import APIError, OpenAI
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

# Load environment variables from a local .env file if present.
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set. Add it to your environment or .env file.")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "user_requests.sqlite3")


def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            idea_name TEXT,
            payload TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


DB_CONN = init_db(DB_PATH)
DB_LOCK = Lock()


def _save_user_request(user_id: str, responses: Dict[str, str]) -> None:
    if not user_id or not responses:
        return
    idea_name = (responses.get("idea_name") or "").strip() or None
    payload = json.dumps(responses, ensure_ascii=False)
    with DB_LOCK:
        DB_CONN.execute(
            "INSERT INTO user_requests (user_id, idea_name, payload) VALUES (?, ?, ?)",
            (user_id, idea_name, payload),
        )
        DB_CONN.commit()


def _fetch_user_requests(user_id: str) -> List[Dict[str, Any]]:
    if not user_id:
        return []
    with DB_LOCK:
        rows = DB_CONN.execute(
            """
            SELECT id, created_at, idea_name, payload
            FROM user_requests
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (user_id,),
        ).fetchall()
    entries: List[Dict[str, Any]] = []
    for row in rows:
        payload_raw = row[3]
        try:
            responses = json.loads(payload_raw) if payload_raw else {}
        except json.JSONDecodeError:
            responses = {}
        entries.append(
            {
                "id": row[0],
                "created_at": row[1],
                "idea_name": row[2] or responses.get("idea_name") or "Не указано",
                "responses": responses,
            }
        )
    return entries


async def store_user_request(user_id: str, responses: Dict[str, str]) -> None:
    await asyncio.to_thread(_save_user_request, user_id, dict(responses))


async def get_user_requests(user_id: str) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(_fetch_user_requests, user_id)

QUESTION_FLOW: List[Tuple[str, str]] = [
    ("idea_name", "Как называется ваш стартап?"),
    ("problem", "В чем основная проблема, которую вы решаете?"),
    ("solution", "Опишите ваше решение и ключевую функциональность."),
    ("audience", "Кто целевая аудитория и пользователи решения?"),
    ("differentiation", "Чем вы отличаетесь от конкурентов или альтернатив?"),
    ("business_model", "Как вы планируете зарабатывать деньги?"),
    ("stage", "На какой стадии находится проект сейчас (идея, прототип, пилот и т.д.)?"),
    ("needs", "Какая поддержка или ресурсы вам нужны?"),
    ("extra", "Есть ли дополнительная информация, которой хотите поделиться?"),
]

STATE_IDS = list(range(len(QUESTION_FLOW)))


def build_idea_summary(responses: Dict[str, str]) -> str:
    """Create a plain-text summary to feed into the OpenAI prompt."""
    lines = []
    for key, question in QUESTION_FLOW:
        value = responses.get(key, "Не указано").strip()
        lines.append(f"{question}\n{value if value else 'Не указано'}")
    return "\n\n".join(lines)


def format_feedback(feedback: Dict[str, Any]) -> str:
    """Format the structured feedback returned by OpenAI for Telegram output."""
    summary = feedback.get("summary", "Нет сводки")
    scores = feedback.get("scores", {})
    overall = feedback.get("overall_rating", {})
    actions = feedback.get("action_items", [])

    lines: List[str] = ["Оценка вашей идеи"]
    lines.append("")
    lines.append(f"Сводка: {summary}")
    lines.append("")
    if isinstance(scores, dict) and scores:
        lines.append("Оценка по критериям:")
        for criterion, data in scores.items():
            if isinstance(data, dict):
                score = data.get("score")
                reason = data.get("reason", "")
                criterion_label = criterion.replace("_", " ").capitalize()
                if score is not None:
                    lines.append(f"- {criterion_label}: {score}/10")
                if reason:
                    lines.append(f"  {reason}")
        lines.append("")
    if isinstance(actions, list) and actions:
        lines.append("Что можно улучшить:")
        for idx, item in enumerate(actions, start=1):
            lines.append(f"{idx}. {item}")
        lines.append("")
    if isinstance(overall, dict) and overall:
        score = overall.get("score")
        verdict = overall.get("verdict", "")
        if score is not None:
            lines.append(f"Итоговая оценка: {score}/10")
        if verdict:
            lines.append(verdict)
    return "\n".join(lines)


def request_feedback_from_openai(responses: Dict[str, str]) -> Dict[str, Any]:
    """Send collected answers to OpenAI and return structured feedback."""
    idea_summary = build_idea_summary(responses)

    system_prompt = (
        "You are an experienced startup advisor. Analyze the provided startup idea summary "
        "and deliver constructive, actionable feedback. Respond in Russian."
    )

    user_prompt = (
        "Проанализируй следующую информацию о стартапе и верни JSON с полями: "
        "summary (строка), scores (объект с ключами problem_clarity, market_potential, "
        "solution_feasibility, competitive_advantage, monetization_strategy и значениями "
        "объектов {score: 0-10, reason: строка}), overall_rating (объект с полями score и verdict), "
        "action_items (список из 3-5 рекомендаций). Используй только валидный JSON.\n\n"
        f"Информация:\n{idea_summary}"
    )

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_output_tokens=600,
        text={"format": {"type": "json_object"}},
    )

    try:
        content = getattr(response, "output_text", None)
        if not content:
            chunks: List[str] = []
            for item in getattr(response, "output", []) or []:
                for content_item in getattr(item, "content", []) or []:
                    text = getattr(content_item, "text", None)
                    if text:
                        chunks.append(text)
            content = "".join(chunks)
        if not content:
            raise RuntimeError("Пустой ответ от модели. Попробуйте еще раз позже.")
        return json.loads(content)
    except json.JSONDecodeError as err:
        logger.error("Failed to parse OpenAI response: %s", err)
        raise RuntimeError("Не удалось обработать ответ от модели. Попробуйте еще раз позже.") from err


async def generate_feedback(responses: Dict[str, str]) -> Dict[str, Any]:
    """Wrap the OpenAI call in a thread to avoid blocking the event loop."""
    try:
        return await asyncio.to_thread(request_feedback_from_openai, responses)
    except APIError as err:
        logger.exception("OpenAI API error: %s", err)
        detail = getattr(err, "message", None)
        if not detail:
            response = getattr(err, "response", None)
            if response is not None:
                json_method = getattr(response, "json", None)
                if callable(json_method):
                    try:
                        payload = json_method()
                    except Exception:  # noqa: BLE001 - best-effort diagnostics
                        payload = None
                    if isinstance(payload, dict):
                        detail = (
                            payload.get("error")
                            or payload.get("message")
                            or payload.get("error_message")
                        )
        if not detail:
            detail = str(err) or None
        detail_text = f": {detail}" if detail else ""
        raise RuntimeError(
            "Ошибка OpenAI API" + detail_text + ". Проверьте ключ, модель и повторите попытку."
        ) from err
    except Exception as err:  # noqa: BLE001 - provide friendly message to the user
        logger.exception("Unexpected error while requesting feedback: %s", err)
        raise RuntimeError(str(err)) from err


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    context.user_data["responses"] = {}
    context.user_data["state"] = 0
    await update.message.reply_text(
        "Привет! Я помогу оценить вашу стартап-идею. Ответьте на несколько вопросов.\n\n"
        + QUESTION_FLOW[0][1]
    )
    return STATE_IDS[0]


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text("Диалог завершен. Можете снова отправить /start, когда будете готовы.")
    return ConversationHandler.END


async def collect_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    state = context.user_data.get("state", 0)
    responses: Dict[str, str] = context.user_data.setdefault("responses", {})

    key, _ = QUESTION_FLOW[state]
    responses[key] = update.message.text.strip()

    next_state = state + 1
    if next_state >= len(QUESTION_FLOW):
        user = update.effective_user
        if user:
            try:
                await store_user_request(str(user.id), responses)
            except Exception as err:  # noqa: BLE001 - logging for diagnostics
                logger.exception("Failed to record request for user %s: %s", user.id, err)
        await update.message.reply_text("Спасибо! Анализирую идею, это может занять несколько секунд...")
        try:
            feedback = await generate_feedback(responses)
            formatted = format_feedback(feedback)
            await update.message.reply_text(formatted)
        except RuntimeError as err:
            await update.message.reply_text(str(err))
        finally:
            context.user_data.clear()
        return ConversationHandler.END

    context.user_data["state"] = next_state
    await update.message.reply_text(QUESTION_FLOW[next_state][1])
    return STATE_IDS[next_state]


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Используйте /start, чтобы начать новый опрос, /history — чтобы посмотреть прошлые запросы, "
        "или /cancel для завершения текущего диалога."
    )


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    message = update.message
    if not user or not message:
        return

    requests = await get_user_requests(str(user.id))
    if not requests:
        await message.reply_text("История запросов пуста. Отправьте /start, чтобы начать.")
        return

    lines = ["Ваши предыдущие запросы:"]
    max_entries = 10
    for idx, entry in enumerate(requests[:max_entries], start=1):
        idea_name = entry.get("idea_name", "Не указано")
        created_at = entry.get("created_at", "")
        lines.append(f"{idx}. {created_at} — {idea_name}")

    if len(requests) > max_entries:
        lines.append(f"Показаны последние {max_entries} из {len(requests)} записей.")

    await message.reply_text("\n".join(lines))


def main() -> None:
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            state: [MessageHandler(filters.TEXT & ~filters.COMMAND, collect_answer)]
            for state in STATE_IDS
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        allow_reentry=True,
    )

    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("history", history_command))

    logger.info("Bot is starting...")
    application.run_polling()


if __name__ == "__main__":
    main()
