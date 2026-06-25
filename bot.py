import asyncio
import logging
import os
import re
import shutil
import tempfile

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import FSInputFile, Message
from dotenv import load_dotenv

from agent import run_analysis

load_dotenv()

#Настройки из .env
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "20"))
MAX_INSTRUCTION_CHARS = int(os.getenv("MAX_INSTRUCTION_CHARS", "2000"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "3"))

#Защита: разрешённые форматы файла и эвристика prompt-injection (RU/EN)
ALLOWED_EXT = {".csv", ".tsv", ".xlsx", ".xls", ".xlsm"}

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|the\s+)*(previous|prior|above)",
    r"disregard\s+(all\s+|the\s+)*(previous|prior|above)",
    r"forget\s+(all\s+|the\s+)*(previous|prior|everything)",
    r"system\s+prompt", r"developer\s+(message|prompt)",
    r"you\s+are\s+now", r"\bact\s+as\b", r"pretend\s+to\s+be",
    r"reveal\s+(your\s+)?(system\s+)?(prompt|instructions)",
    r"print\s+(your\s+)?(system\s+)?(prompt|instructions)",
    r"\bjailbreak\b", r"new\s+instructions?:",
    r"игнорир", r"забудь\s+(все|всё|вс[её]\s+)?(предыдущ|прошл)",
    r"не\s+обращай\s+внимани", r"нов(ая|ые)\s+инструкци",
    r"ты\s+теперь", r"представь\s+что\s+ты", r"веди\s+себя\s+как",
    r"систем(ный|ное)\s+(промпт|сообщени)",
    r"покажи\s+(свой\s+)?(систем|промпт|инструкц)",
    r"раскрой\s+(свой\s+)?промпт",
    r"rm\s+-rf", r"os\.system", r"\bsubprocess\b", r"__import__",
    r"\beval\(", r"\bexec\(", r"exfiltrat", r"\bcurl\b", r"\bwget\b",
    r"requests\.(get|post)",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def validate_file(filename, size_bytes):
    #Проверяем формат и размер файла до загрузки
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        return False, (f"Неподдерживаемый формат: {ext or '-'}. "
                       "Поддерживаются CSV, TSV, XLSX, XLS.")
    if size_bytes and size_bytes > MAX_FILE_MB * 1024 * 1024:
        return False, (f"Файл слишком большой ({size_bytes / 1e6:.1f} МБ). "
                       f"Лимит - {MAX_FILE_MB} МБ.")
    return True, ""


def sanitize_instruction(text):
    #Обрезаем инструкцию по длине и прогоняем через эвристику prompt-injection.
    #Возвращаем (очищенный_текст, подозрительно, список_совпадений).
    if not text:
        return "", False, []
    clean = text.strip()[:MAX_INSTRUCTION_CHARS]
    matched = [p.pattern for p in _COMPILED if p.search(clean)]
    return clean, bool(matched), matched


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("bot")

#Без токенов запускаться нет смысла
if not TELEGRAM_TOKEN or not OPENROUTER_API_KEY:
    raise SystemExit("Не заданы TELEGRAM_BOT_TOKEN и/или OPENROUTER_API_KEY. "
                     "Скопируй .env.example в .env и заполни значения.")

bot = Bot(TELEGRAM_TOKEN)
dp = Dispatcher()
analysis_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
pending_instruction = {}  #chat_id -> инструкция, присланная отдельным сообщением до файла

WELCOME = (
    "Привет! Я - ИИ-аналитик данных.\n\n"
    "Пришли датасет (CSV, TSV или Excel), и я как агент сам исследую его: "
    "посчитаю метрики, найду инсайды, построю графики и пришлю отчёт.\n\n"
    "Можешь добавить инструкцию - что важно (подписью к файлу или сообщением "
    "перед ним). \n\n/help - подробнее."
)

HELP = (
    "Как пользоваться:\n"
    f"1) Пришли файл CSV/TSV/XLSX/XLS (до {MAX_FILE_MB} МБ).\n"
    "2) По желанию добавь инструкцию - подписью к файлу или сообщением перед ним.\n"
    "3) Подожди пару минут - пришлю отчёт и графики.\n\n"
    "Внутри: модель не пересказывает готовые цифры, а сама пишет и выполняет "
    "Python в песочнице (pandas, matplotlib) и итеративно строит анализ.\n\n"
    "Защита: инструкция и содержимое файла считаются недоверенными "
    "данными; попытки prompt-injection отслеживаются, код выполняется без "
    "доступа к секретам бота."
)


@dp.message(CommandStart())
async def on_start(message: Message):
    await message.answer(WELCOME)


@dp.message(Command("help"))
async def on_help(message: Message):
    await message.answer(HELP)


@dp.message(F.document)
async def on_document(message: Message):
    #Главный обработчик: пришёл файл - проверяем, скачиваем, анализируем
    doc = message.document
    ok, err = validate_file(doc.file_name, doc.file_size)
    if not ok:
        await message.answer(err)
        return

    instruction = message.caption or pending_instruction.pop(message.chat.id, "")
    clean_instr, suspicious, matched = sanitize_instruction(instruction)
    if suspicious:
        log.warning("Возможная prompt-injection: %s", matched)

    session_dir = tempfile.mkdtemp(prefix="analysis_")
    data_path = os.path.join(session_dir, doc.file_name)
    try:
        await bot.download(doc, destination=data_path)
    except Exception as exc:
        await message.answer(f"Не удалось скачать файл: {exc}")
        shutil.rmtree(session_dir, ignore_errors=True)
        return

    prefix = ""
    if suspicious:
        prefix = ("В инструкции обнаружены признаки prompt-injection - они "
                  "проигнорированы, анализ проведён строго по данным.\n\n")

    status = await message.answer("Анализирую данные… это может занять пару минут.")
    try:
        #агент синхронный, поэтому уводим его в поток, чтобы не блокировать бота
        async with analysis_semaphore:
            result = await asyncio.to_thread(
                run_analysis, data_path, session_dir, clean_instr, suspicious)
        await _send_report(message, prefix + result["report"], result["images"])
        log.info("Готово за %s шаг(ов) агента, графиков: %s",
                 result["iterations"], len(result["images"]))
    except Exception as exc:
        log.exception("Ошибка анализа")
        await message.answer(f"Произошла ошибка при анализе: {exc}")
    finally:
        try:
            await status.delete()
        except Exception:
            pass
        shutil.rmtree(session_dir, ignore_errors=True)

@dp.message(F.text & ~F.text.startswith("/"))
async def on_text(message: Message):
    #Текст без файла запоминаем как инструкцию к следующему файлу
    pending_instruction[message.chat.id] = message.text
    await message.answer("Принял инструкцию. Теперь пришли файл (CSV/Excel) - "
                         "проанализирую его с учётом твоего запроса.")


async def _send_report(message: Message, report, images):
    #Отчёт режем на куски (лимит Telegram), графики шлём отдельными фото
    for chunk in _split(report, 4000):
        await message.answer(chunk)
    for image in images:
        try:
            caption = (image.get("title") or "")[:1024] or None
            await message.answer_photo(FSInputFile(image["path"]), caption=caption)
        except Exception as exc:
            log.warning("Не удалось отправить график %s: %s", image.get("path"), exc)


def _split(text, size):
    text = text or "(пустой отчёт)"
    return [text[i:i + size] for i in range(0, len(text), size)]


async def main():
    log.info("Бот запускается, модель из agent.MODEL")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
