import os
import logging
import subprocess
import openai
from transformers import pipeline, GPT2Tokenizer
from concurrent.futures import ThreadPoolExecutor
from telegram import ext, Bot

# Set tokens and keys from environment variables
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SYSTEM_MESSAGE = os.getenv('SYSTEM_MESSAGE')

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

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def truncate_text_to_tokens(text, max_tokens):
    if len(tokenizer.encode(text)) > max_tokens:
        tokens = tokenizer.tokenize(text)
        while len(tokenizer.encode(' '.join(tokens[-max_tokens:]))) > max_tokens:
            tokens.pop(0)
        return ' '.join(tokens[-max_tokens:])
    return text

# Function to process the message with OpenAI
def process_message_with_openai(message_text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": message_text}
        ],
        temperature=0.5
    )
    return response

# Main function for handling user messages
def handle_message(update, context, simulated_message=None):
    try:
        # Check if the user is allowed to send messages
        user_nickname = update.message.from_user.username if not simulated_message else None
        if user_nickname and user_nickname not in ALLOWED_USERS:
            logging.warning(f"User {user_nickname} is not allowed to use the bot")
            return

        # Get the message text from the user or the simulated message
        message_text = simulated_message if simulated_message else update.message.text
        logging.debug(f"Received message: {message_text}")

        # Use global ThreadPoolExecutor to process the message with OpenAI
        response = executor.submit(process_message_with_openai, message_text).result()

        logging.debug(f"Request sent to OpenAI for processing")

        # Get the response from OpenAI
        response_text = response.choices[0].message.content.strip()
        logging.debug(f"Received response from OpenAI: {response_text}")

        # Send the response to the user
        chat_id = update.message.chat_id
        bot.send_message(chat_id=chat_id, text=response_text)
        logging.debug(f"Sent response to user {chat_id}: {response_text}")

        if "<!EXECUTE>" in response_text:
            # Execute a command
            command_start = response_text.find("<!EXECUTE>") + len("<!EXECUTE>")
            command_end = response_text.find("</EXECUTE>")
            command = response_text[command_start:command_end].strip()

            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
            command_result, command_error = process.communicate()

            response_text = response_text.replace("<!EXECUTE>", "").replace("</EXECUTE>", "").strip()
            response_text += f"\n\nOUTPUT: {command_result}"
            if command_error:
                response_text += f"\n\nERROR: {command_error}"
                logging.debug(f"Executed command {command} with result {command_result}, error {command_error}")
            else:
                logging.debug(f"Executed command {command} with result {command_result}")

            # Send the response to the user
            chat_id = update.message.chat_id
            bot.send_message(chat_id=chat_id, text=response_text)
            logging.debug(f"Sent response to user {chat_id}: {response_text}")

            # Send the command result as a new message
            simulated_message = "The command output is:\n\n" + command_result
            if command_error:
                simulated_message += "\n\nThe command error is:\n\n" + command_error

            # Truncate the message to the last 2048 tokens
            simulated_message = truncate_text_to_tokens(simulated_message, 2048)

            handle_message(update, context, simulated_message=simulated_message)

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

        # Create global ThreadPoolExecutor
        executor = ThreadPoolExecutor()

        # Start the bot
        updater.start_polling()
        logging.info("Telegram bot is running and ready to work")

        # Keep the main thread running
        updater.idle()

    except Exception as e:
        logging.error(f"An error occurred while running the Telegram bot: {e}")
    finally:
        # Clean up the ThreadPoolExecutor
        executor.shutdown()
