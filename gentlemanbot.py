import configparser
import smtplib
from email.mime.text import MIMEText
from openai import OpenAI
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone
from typing import Optional, Dict

config = configparser.ConfigParser()
config.read('config.ini')

client = OpenAI(api_key=config['OpenAI']['api_key'])
from aiohttp import web
import asyncio
import aiocron

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# OpenAI configuration
model = config['OpenAI'].get('model', 'gpt-4o')

# SMTP configuration
smtp_server = config['SMTP']['smtp_server']
smtp_port = int(config['SMTP']['smtp_port'])
smtp_email = config['SMTP']['email']
smtp_password = config['SMTP']['password']

# SMS recipient configuration
recipient_carrier_gateway = config['SMS']['recipient_carrier_gateway']
recipient_number = config['SMS']['recipient_number']

def connect_to_mongodb(self):
        try:
            if self.mongo_connection_uri:
                logger.debug(f"Connecting with URI: {self.mongo_connection_uri}")
                self.mongo_client = MongoClient(self.mongo_connection_uri)
            else:
                self.mongo_client = MongoClient(
                    host=self.mongo_host,
                    port=self.mongo_port,
                    username=self.mongo_username,
                    password=self.mongo_password,
                    directConnection=True)

            self.mongo_db = self.mongo_client[self.mongo_db]
            self.event_collection = self.mongo_db[self.mongo_collection]
        except ConnectionFailure as e:
            logger.exception(f"Could not connect to MongoDB: {e}")
            raise

# MongoDB setup
# mongo_client = AsyncIOMotorClient(config['MongoDB']['connection_uri'])
mongo_client = AsyncIOMotorClient(
    host=config['MongoDB']['host'],
    port=int(config['MongoDB']['port']),
    username=config['MongoDB']['username'],
    password=config['MongoDB']['password'],
    directConnection=True
)
mongo_db = mongo_client[config['MongoDB']['db']]
users_collection = mongo_db[config['MongoDB']['collection']]

# User model class
class UserPreferences:
    def __init__(self, phone_number: str, _id=None, **kwargs):
        self.phone_number = phone_number
        self.system_prompt: str = kwargs.get('system_prompt', "You are a gentleman bot who sends polite and motivational messages.")
        self.morning_prompt: str = kwargs.get('morning_prompt', "Write a positive and motivational message to make someone's day better.")
        self.personal_info: Dict = kwargs.get('personal_info', {})
        self.created_at: datetime = kwargs.get('created_at', datetime.now(timezone.utc))
        self.updated_at: datetime = kwargs.get('updated_at', datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "phone_number": self.phone_number,
            "system_prompt": self.system_prompt,
            "morning_prompt": self.morning_prompt,
            "personal_info": self.personal_info,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

# User management functions
async def get_or_create_user(phone_number: str) -> UserPreferences:
    # Exclude _id field from the query result
    user_data = await users_collection.find_one(
        {"phone_number": phone_number},
        {'_id': 0}  # Exclude _id field
    )
    if not user_data:
        user = UserPreferences(phone_number)
        await users_collection.insert_one(user.to_dict())
        return user
    return UserPreferences(**user_data)

async def update_user_preferences(phone_number: str, updates: Dict) -> None:
    updates["updated_at"] = datetime.now(timezone.utc)
    await users_collection.update_one(
        {"phone_number": phone_number},
        {"$set": updates}
    )

# Function to generate a message using OpenAI
async def generate_message(incoming_message: str, phone_number: str):
    user = await get_or_create_user(phone_number)
    
    # Add personal context if available
    context = ""
    if user.personal_info:
        context = "Here's some context about the user: " + str(user.personal_info)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": user.system_prompt},
            {"role": "system", "content": context},
            {"role": "user", "content": incoming_message}
        ],
        max_tokens=50
    )
    return response.choices[0].message.content.strip()

# Function to send SMS via SMTP
async def send_sms(message):
    msg = MIMEText(message)
    msg['Subject'] = "Gentleman Bot Message"
    msg['From'] = smtp_email
    msg['To'] = f"{recipient_number}@{recipient_carrier_gateway}"

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_email, smtp_password)
            server.sendmail(smtp_email, recipient_carrier_gateway, msg.as_string())
            print(f"Message sent: {message}")
    except Exception as e:
        print(f"Error sending message: {e}")

# Schedule a daily message at 8:00 AM
@aiocron.crontab('0 8 * * *')  # Runs at 8:00 AM every day
async def daily_message():
    print("Generating and sending daily messages...")
    user = await get_or_create_user(recipient_number)
    message = await generate_message(user.morning_prompt, recipient_number)
    await send_sms(message)

# Webhook handler for incoming messages
async def webhook_handler(request):
    form_data = await request.post()
    incoming_message = form_data.get('Body', '').strip()
    sender_number = form_data.get('From', '').strip()
    
    print(f"Received message from {sender_number}: {incoming_message}")
    
    response_message = await generate_message(incoming_message, sender_number)
    await send_sms(response_message)
    
    return web.Response(text="OK")

# Main application setup
app = web.Application()
app.router.add_post('/webhook', webhook_handler)

async def main():
    print("Gentleman Bot is running...")
    # Start the cron job for daily messaging
    # asyncio.create_task(daily_message())

    default_message = "Write a positive and motivational message to make someone's day better."
    message = await generate_message(default_message, recipient_number)
    print(f"Initial message: {message}")

    # Run the web server
    # return await web.run_app(app, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    asyncio.run(main())
