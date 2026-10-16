import argparse
import signal
import sys
from collections.abc import Callable

import groq
import httpx
from pydantic import ValidationError

from react_lite.agent import Agent
from react_lite.config import Settings
from react_lite.console import ActionPrinter
from react_lite.events import Events, file_logger
from react_lite.llm import GroqModel
from react_lite.tools import ExchangeRates, Tools

HELP = "/help — помощь; /reset — удалить контекст; /exit — выход. Ctrl+C и Ctrl+D — выход."


def interact(
    agent: Agent,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> None:
    write("CLI-агент ReAct-lite. " + HELP)
    while True:
        try:
            user = read("Вы: ").strip()
            if not user:
                continue
            if user == "/exit":
                break
            if user == "/help":
                write(HELP)
            elif user == "/reset":
                agent.reset()
                write("Контекст диалога удалён.")
            elif user.startswith("/"):
                write("Неизвестная команда. " + HELP)
            else:
                write("Бот: " + agent.run(user))
        except (KeyboardInterrupt, EOFError):
            break
    write("До свидания!")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CLI-агент с планированием через Groq API")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Короткие шаги действий (также включены по умолчанию); подробности — в JSONL-файле",
    )
    parser.parse_args(argv)
    if not hasattr(signal, "setitimer"):
        print("Требуется POSIX-система (Linux/macOS) для общего дедлайна.", file=sys.stderr)
        return 2
    try:
        settings = Settings()
    except ValidationError as exc:
        fields = ", ".join(".".join(map(str, issue["loc"])) for issue in exc.errors())
        print(f"Ошибка настроек: проверьте {fields} в окружении или .env.", file=sys.stderr)
        return 2
    try:
        logger = file_logger(settings.log_file)
    except OSError:
        print("Не удалось открыть лог. Проверьте LOG_FILE и права записи.", file=sys.stderr)
        return 2
    key = settings.groq_api_key.get_secret_value().strip()
    if not key:
        print("Ключ Groq не настроен. Добавьте GROQ_API_KEY в локальный .env.")
    print(f"Модель: {settings.groq_model}")
    events = Events(logger, key, ActionPrinter())
    # Клиент без ключа не создаём: CLI и команды доступны и при пустой конфигурации.
    client = groq.Groq(api_key=key, max_retries=0) if key else None
    try:
        with httpx.Client(follow_redirects=True) as http:
            model = GroqModel(
                client,
                settings.groq_model,
                settings.network_timeout_seconds,
                settings.network_attempts,
            )
            tools = Tools(
                ExchangeRates(
                    http,
                    settings.network_timeout_seconds,
                    settings.network_attempts,
                )
            )
            interact(
                Agent(
                    model,
                    tools,
                    events,
                    settings.max_steps,
                    settings.max_tool_calls,
                    settings.request_timeout_seconds,
                )
            )
    finally:
        if client:
            client.close()
        for handler in logger.handlers:
            handler.flush()
    return 0
