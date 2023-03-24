import os
import logging
import json
import asyncio
import openai
import tiktoken
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from cachetools import TTLCache

# Set tokens and keys from environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

# Set logging level to DEBUG
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.DEBUG
)

# Initialize Telegram bot
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())


# Initialize MongoDB
client = AsyncIOMotorClient(MONGO_DB_URL)
db = client["telegram_bot_db"]
conversations = db["conversations"]
tokens_collection = db["tokens"]

# Create an index for chat_id if it does not already exist
async def create_chat_id_index():
    await conversations.create_index("chat_id")

asyncio.get_event_loop().run_until_complete(create_chat_id_index())


# Create a cache with a lifetime of 20 minutes
cache = TTLCache(maxsize=1024, ttl=1200)

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

async def truncate_msgs_to_tokens(messages, token_limit):
    logging.debug("Truncuate messages")

    msg_encoded = enc.encode(json.dumps(messages[-1]))
    msg_length = len(msg_encoded)
    if msg_length > token_limit:
        raise ValueError(
            f"The message '{messages[-1]['content']}' contains {msg_length} tokens, which exceeds the limit ({token_limit})"
        )

    messages_encoded = enc.encode(json.dumps(messages))
    total_length = len(messages_encoded)
    while total_length > token_limit:
        messages.pop(0)
        messages_encoded = enc.encode(json.dumps(messages))
        total_length = len(messages_encoded)

    return messages


# Handler for the /start command
@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    try:
        await message.reply("Hello! I'm ready to work.")
        logging.info(f"User {message.chat.id} started the bot")
    except Exception as e:
        logging.error(f"An error occurred while processing the /start command: {e}")


@dp.message_handler(commands=['token'])
async def get_openai_api_token(message: types.Message):
    if len(message.text.split()) != 2:
        await message.reply("Please make sure that you sent the command in the format /token <your_openai_api_token>.")
        return

    openai_api_token = message.text.split()[1]
    chat_id = message.chat.id

    # Save a token in the MongoDB collection.
    await tokens_collection.update_one(
        {'chat_id': chat_id},
        {'$set': {'openai_api_token': openai_api_token}},
        upsert=True
    )

    try:
        await message.reply("Token saved.")
        logging.info(f"User {message.chat.id} started the bot")
    except Exception as e:
        logging.error(f"An error occurred while processing the /token command: {e}")


# Function to process the message with OpenAI
async def process_message_with_openai(
    msgs, token_limit=3000, response_token_limit=1000
):
    logging.debug(f"Trying to send messages to ChatGPT")

    try:
        truncated_msgs = await truncate_msgs_to_tokens(msgs, token_limit)
    except ValueError as e:
        logging.error(e)
        return f"Error: {e}"

    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        max_tokens=response_token_limit,
        messages=truncated_msgs,
        temperature=0.5,
        presence_penalty=0.5,
        frequency_penalty=0.5,
    )
    return response


# Load messages
async def load_messages(chat_id):
    if chat_id in cache:
        return cache[chat_id]

    messages = list(await conversations.find({"chat_id": chat_id}).sort("_id", -1).limit(300).to_list(None))

    oldest_message_id = messages[-1]["_id"] if len(messages) > 0 else None

    if oldest_message_id:
        await conversations.delete_many({"chat_id": chat_id, "_id": {"$lt": oldest_message_id}})
        logging.debug(f"Deleted old messages for chat {chat_id}")

    result = [message["message"] for message in messages] if len(messages) > 0 else []

    cache[chat_id] = result
    return result


# Add message into db and update cache
async def save_messages(chat_id, messages, token: str=None):
    for message in messages:
        await conversations.update_one(
            {"chat_id": chat_id, "message": message},
            {"$set": {
                "chat_id": chat_id,
                "message": message
            }},
            upsert=True,
        )

    if chat_id in cache:
        cached_messages = cache[chat_id]
        for message in messages:
            if message not in cached_messages:
                cached_messages.append(message)


# Main function for handling user messages
async def handle_message(message: types.Message):
    try:
        message_text = message.text
        chat_id = message.chat.id

        # Initialize OpenAI API
        token_entry = await tokens_collection.find_one({"chat_id": chat_id})
        if not token_entry:
            await message.reply("OpenAI API token not found for this chat.")
            return
        openai.api_key = token_entry['openai_api_token']

        # Load context
        messages = await load_messages(chat_id)
        messages.append(
            {"role": "user", "content": message_text}
        )

        # Process the message with OpenAI
        response = await process_message_with_openai(messages)
        logging.debug(f"Request sent to OpenAI for processing")

        # Get the response from OpenAI
        response_text = response.choices[0].message.content.strip()
        if response_text:
            # Save the conversation with the new response
            messages.append(
                {"role": "assistant", "content": response_text}
            )
            await save_messages(chat_id, messages[-2:])

            # Send the response to the user
            await bot.send_message(chat_id=chat_id, text=response_text)
            logging.debug(f"Sent response to user {chat_id}")
        else:
            logging.debug(f"Received empty response from OpenAI")

    except Exception as e:
        logging.error(f"An error occurred while processing the user message: {e}")
        try:
            await bot.send_message(chat_id=chat_id, text="Sorry, something went wrong while processing your request. Please try again.")
            logging.debug(f"Sent error message to user {chat_id}")
        except Exception as e:
            logging.error(f"An error occurred while trying to reconnect to Telegram bot: {e}")

dp.register_message_handler(handle_message, content_types=types.ContentTypes.TEXT)

# Main program loop
if __name__ == '__main__':
    try:
        logging.info("Telegram bot is running and ready to work")
        
        # Start the bot
        from aiogram import executor
        executor.start_polling(dp, skip_updates=True)
        
    except Exception as e:
        logging.error(f"An error occurred while running the Telegram bot: {e}")
    finally:
        # Close MongoDB connection
        client.close()
