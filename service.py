import json
import os
import logging
import subprocess
import openai
from transformers import GPT2Tokenizer
from telegram import ext, Bot

# Set tokens and keys from environment variables
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MONGO_DB_URL = os.getenv('MONGO_DB_URL')

# Initialize Telegram bot
bot = Bot(token=TELEGRAM_TOKEN)

# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

# Initialize mongoDB
client = MongoClient(MONGO_DB_URL)
db = client['telegram_bot_db']
context_collection = db['context_collection']

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


def truncate_msgs_to_tokens(messages, token_limit):
    for message in messages:
        tokenized_message = tokenizer(message["content"], return_tensors="pt")
        message_tokens = tokenized_message["input_ids"].shape[1]
        if message_tokens > token_limit:
            raise ValueError(f"The message '{message['content']}' contains {message_tokens} tokens, which exceeds the limit ({token_limit}).")

    tokenized_messages = tokenizer(messages, return_tensors="pt")
    total_tokens = tokenized_messages["input_ids"].shape[1]

    while total_tokens > token_limit:
        messages.pop(0)
        tokenized_messages = tokenizer(messages, return_tensors="pt")
        total_tokens = tokenized_messages["input_ids"].shape[1]

    return messages


# Function to process the message with OpenAI
def process_message_with_openai(msgs, token_limit=2048, response_token_limit=1024):
    logging.debug(f"Trying to send messages to ChatGPT.")

    try:
        truncated_msgs = truncate_msgs_to_tokens(msgs, token_limit)
    except ValueError as e:
        logging.error(e)
        return f"Error: {e}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=response_token_limit,
        messages=truncated_msgs,
        temperature=0.66,
        presence_penalty=0.66,
        frequency_penalty=0.66
    )
    return response


# Main function for handling user messages
def handle_message(update, context):
    try:
        # Check if the user is allowed to send messages
        user_nickname = update.message.from_user.username
        if user_nickname and user_nickname not in ALLOWED_USERS:
            logging.warning(f"User {user_nickname} is not allowed to use the bot")
            return

        message_text = update.message.text

        # Get current user context from the database
        chat_id = update.message.chat_id
        context_data = context_collection.find_one({'chat_id': chat_id})

        # Prepare messages for OpenAI with current user context
        messages = []
        if context_data:
            messages.append({"role": "user", "content": context_data['context']})
        messages.append({"role": "user", "content": message_text})

        # Process the message with OpenAI
        response = process_message_with_openai(messages)
        logging.debug(f"Request sent to OpenAI for processing")

        # Get the response from OpenAI
        response_text = response.choices[0].message.content.strip()
        if response_text:
            # Save current user context to the database
            context_collection.update_one({'chat_id': chat_id}, {'$set': {'context': json.dumps([messages[-1], {"role": "ai_assistance", "content": response_text}])}}, upsert=True)

            # Send the response to the user
            chat_id = update.message.chat_id
            bot.send_message(chat_id=chat_id, text=response_text)
            logging.debug(f"Sent response to user {chat_id}: {response_text}")
        else:
            logging.debug(f"Received empty response from OpenAI")

    except Exception as e:
        logging.error(f"An error occurred while processing the user message: {e}")
        try:
            # Try to reconnect and send the message again
            bot = Bot(token=TELEGRAM_TOKEN)
            bot.send_message(chat_id=chat_id, text="Sorry, something went wrong while processing your request. Please try again.")
            logging.debug(f"Sent error message to user {chat_id}")
        except Exception as e:
            logging.error(f"An error occurred while trying to reconnect to Telegram bot: {e}")

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

        # Keep the main thread running
        updater.idle()

    except Exception as e:
        logging.error(f"An error occurred while running the Telegram bot: {e}")
