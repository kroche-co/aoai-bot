import time
import json
import os
import logging
import openai
from transformers import GPT2Tokenizer
from telegram import ext, Bot
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from cachetools import TTLCache
from transliterate import translit


# Set tokens and keys from environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_DB_URL = os.getenv("MONGO_DB_URL")


# Set logging level to DEBUG
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.DEBUG
)


# Initialize Telegram bot
bot = Bot(token=TELEGRAM_TOKEN)


# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY


# Initialize mongoDB
client = MongoClient(MONGO_DB_URL)
try:
    # The ismaster command is cheap and does not require auth.
    client.admin.command("ismaster")
except ConnectionFailure:
    logging.error(f"Server not available")

db = client["telegram_bot_db"]
conversations = db["conversations"]

# Create an index for chat_id if it does not already exist
conversations.create_index("chat_id")


# Create a cache with a lifetime of 1 hour
cache = TTLCache(maxsize=1024, ttl=1200)


# Handler for the /start command
def start(update, context):
    try:
        context.bot.send_message(
            chat_id=update.message.chat_id, text="Hello! I'm ready to work."
        )
        logging.info(f"User {update.message.chat_id} started the bot")
    except Exception as e:
        logging.error(f"An error occurred while processing the /start command: {e}")


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def truncate_msgs_to_tokens(messages, token_limit):
    logging.debug(f"Truncuate messages.")

    tokenized_msg = tokenizer.encode(json.dumps(messages[-1]))
    message_tokens = len(tokenized_msg)
    if message_tokens > token_limit:
        raise ValueError(
            f"The message '{messages[-1]['content']}' contains {message_tokens} tokens, which exceeds the limit ({token_limit})."
        )

    tokenized_msgs = tokenizer.encode(json.dumps(messages))
    total_tokens = len(tokenized_msgs)
    while total_tokens > token_limit:
        messages.pop(0)
        tokenized_msgs = tokenizer.encode(json.dumps(messages))
        total_tokens = len(tokenized_msgs)

    return messages


# Function to process the message with OpenAI
def process_message_with_openai(
    msgs, token_limit=2048, response_token_limit=1024
):
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
        frequency_penalty=0.66,
    )
    return response


# Load messages
def load_messages(chat_id):
    # If chat ID is present in the cache, return messages from the cache
    if chat_id in cache:
        return cache[chat_id]

    # Otherwise, retrieve messages from the database
    # Sort them by the "_id" field in descending order (from newest to oldest) and limit the number to 100
    messages = list(conversations.find({"chat_id": chat_id}).sort("_id", -1).limit(300))

    # Get the _id of the oldest message among the last 100
    oldest_message_id = messages[-1]["_id"] if len(messages) > 0 else None

    if oldest_message_id:
        # Delete all messages that are older than the oldest message among the last 100 messages for this chat
        conversations.delete_many({"chat_id": chat_id, "_id": {"$lt": oldest_message_id}})
        logging.debug(f"Deleted old messages for chat {chat_id}")

    # If messages are found, extract their content, otherwise return an empty list
    result = [message["message"] for message in messages] if len(messages) > 0 else []

    # Save the result in the cache with the chat_id as the key
    cache[chat_id] = result
    return result


# Add message into db and update cache
def save_messages(chat_id, messages):
    for message in messages:
        conversations.update_one(
            {"chat_id": chat_id, "message": message},
            {"$set": {"chat_id": chat_id, "message": message, "tags": message.get("tags", None)}},
            upsert=True,
        )

    # Если чат уже находится в кеше, обновите его
    if chat_id in cache:
        cached_messages = cache[chat_id]
        for message in messages:
            if message not in cached_messages:
                cached_messages.append(message)


# Main function for handling user messages
def handle_message(update, context):
    try:
        message_text = update.message.text
        chat_id = update.message.chat_id

        messages = load_messages(chat_id)
        messages.append(
            {"role": "user", "content": translit(message_text, 'ru', reversed=True)}
        )

        # Process the message with OpenAI
        response = process_message_with_openai(messages)
        logging.debug(f"Request sent to OpenAI for processing")

        # Get the response from OpenAI
        response_text = response.choices[0].message.content.strip()
        if response_text:
            # Save the conversation with the new response
            messages.append(
                {"role": "assistant", "content": translit(response_text, 'ru', reversed=True)}
            )
            save_messages(chat_id, messages[-2:])

            # Send the response to the user
            bot.send_message(chat_id=chat_id, text=response_text)
            logging.debug(f"Sent response to user {chat_id}: {response_text}")
        else:
            logging.debug(f"Received empty response from OpenAI")

    except Exception as e:
        logging.error(f"An error occurred while processing the user message: {e}")
        try:
            # Try to reconnect and send the message again
            n_bot = Bot(token=TELEGRAM_TOKEN)
            n_bot.send_message(chat_id=chat_id, text="Sorry, something went wrong while processing your request. Please try again.")
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
    finally:
        # Close MongoDB connection
        client.close()
