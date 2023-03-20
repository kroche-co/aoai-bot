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
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            max_tokens=2048
            messages=[
                # {"role": "system", "content": "You are helping to manage a real K8S cluster and you have the ability to run any commands on it through messages starting with 'EXECUTE:'."},
                {"role": "user", "content": message_text}
            ],
            temperature=0.5
        )
        logging.debug(f"Request sent to OpenAI for processing")

        # Get the response from OpenAI
        response_text = response.choices[0].message.content.strip()
        logging.debug(f"Received response from OpenAI: {response_text}")

        if "<!EXECUTE>" in response_text:
            # Execute a command
            command_start = response_text.find("<!EXECUTE>") + len("<!EXECUTE>")
            command_end = response_text.find("</EXECUTE>")
            command = response_text[command_start:command_end].strip()
            command_result = os.popen(command).read()
            response_text = response_text.replace("<!EXECUTE>", "").replace("</EXECUTE>", "").strip()
            response_text = f"{response_text}\n\nRESPONSE: {command_result}"
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
