import os
import logging
from telegram import ext, Bot
import openai

# Set tokens and keys from environment variables
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize Telegram bot
bot = Bot(token=TELEGRAM_TOKEN)

# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

# Set logging level to DEBUG
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)

# List of allowed user nicknames
ALLOWED_USERS = ['kukaryambik']

# Handler for the /start command
def start(update, context):
    try:
        context.bot.send_message(chat_id=update.message.chat_id, text="Hello! I'm ready to work.")
        logging.info(f"User {update.message.chat_id} started the bot")
    except Exception as e:
        logging.error(f"An error occurred while processing the /start command: {e}")

# Main function for handling user messages
def handle_message(update, context):
    try:
        # Check if the user is allowed to send messages
        user_nickname = update.message.from_user.username
        if user_nickname not in ALLOWED_USERS:
            logging.warning(f"User {user_nickname} is not allowed to use the bot")
            return

        # Get the message text from the user
        message_text = update.message.text
        logging.debug(f"Received message from user {user_nickname}: {message_text}")

        # Send a request to OpenAI
        response = openai.Completion.create(
            prompt=message_text,
            max_tokens=2048,
            n=1,
            stop=None,
            temperature=0.6,
            model="text-davinci-003"
        )
        logging.debug(f"Request sent to OpenAI for processing")

        # Get the response from OpenAI
        response_text = response.choices[0].text.strip()
        logging.debug(f"Received response from OpenAI: {response_text}")

        # Check for the "EXECUTE:" tag
        if response_text.startswith("EXECUTE:"):
            # Execute a command
            command = response_text.replace("EXECUTE:", "").strip()
            command_result = os.popen(command).read()
            response_text = f"RESPONSE: {command_result}"
            logging.debug(f"Executed command {command} with result {command_result}")

        # Send the response to the user
        chat_id = update.message.chat_id
        bot.send_message(chat_id=chat_id, text=response_text)
        logging.debug(f"Sent response to user {chat_id}: {response_text}")

    except Exception as e:
        logging.error(f"An error occurred while processing the user message: {e}")

# Main program loop
if __name__ == '__main__':
    try:
        # Create an object for working with Telegram
        updater = ext.Updater(token=TELEGRAM_TOKEN, use_context=True)

        # Create command handlers
        start_handler = ext.CommandHandler('start', start)

        # Register command handlers
        updater.dispatcher.add_handler(start_handler)

        # Create message handler
        message_handler = ext.MessageHandler(
            ext.Filters.text, handle_message)

        # Register message handler
        updater.dispatcher.add_handler(message_handler)

        # Start the bot
        updater.start_polling()
        logging.info("Telegram bot is running and ready to work")

    except Exception as e:
        logging.error(f"An error occurred while running the Telegram bot: {e}")
