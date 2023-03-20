import os
from telegram import ext, Bot
import openai

# Устанавливаем токены и ключи из переменных окружения
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Инициализируем телеграм-бота
bot = Bot(token=TELEGRAM_TOKEN)

# Инициализируем OpenAI API
openai.api_key = OPENAI_API_KEY

# Обработчик команды /start
def start(update, context):
    context.bot.send_message(chat_id=update.message.chat_id, text="Привет! Я готов к работе.")

# Основная функция для обработки сообщений от пользователя
def handle_message(update, context):
    # Получаем текст сообщения от пользователя
    message_text = update.message.text

    # Отправляем запрос в OpenAI
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=message_text,
        max_tokens=1024,
        n=1,
        stop="\n",
        temperature=0.7,
    )

    # Получаем ответ от OpenAI
    response_text = response.choices[0].text.strip()

    # Проверяем наличие метки "EXECUTE:"
    if response_text.startswith("EXECUTE:"):
        # Выполняем команду
        command = response_text.replace("EXECUTE:", "").strip()
        command_result = os.popen(command).read()
        response_text = f"RESPONSE: {command_result}"

    # Отправляем ответ пользователю
    chat_id = update.message.chat_id
    bot.send_message(chat_id=chat_id, text=response_text)

# Основной цикл программы
if __name__ == '__main__':
    # Создаем объект для работы с телеграмом
    updater = ext.Updater(token=TELEGRAM_TOKEN, use_context=True)

    # Создаем обработчики команд
    start_handler = ext.CommandHandler('start', start)

    # Регистрируем обработчики команд
    updater.dispatcher.add_handler(start_handler)

    # Создаем обработчик сообщений
    message_handler = ext.MessageHandler(
        ext.Filters.text, handle_message)

    # Регистрируем обработчик сообщений
    updater.dispatcher.add_handler(message_handler)

    # Запускаем бота
    updater.start_polling()

    # Запускаем цикл обработки сообщений
    updater.idle()
