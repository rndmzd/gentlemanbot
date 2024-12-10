import configparser
import smtplib
from email.mime.text import MIMEText
from openai import OpenAI
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone
from typing import Optional, Dict, List
import logging
import asyncio
import aiocron
import imaplib
import email
from email.header import decode_header
import pytz
import phonenumbers
from phonenumbers import geocoder, timezone as ph_timezone
import json
import sys

# =======================
# Configuration and Logging
# =======================

config = configparser.ConfigParser()
config.read('config.ini')

# =======================
# Logging Configuration
# =======================

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the desired logging level

# Create handlers
file_handler = logging.FileHandler("gentleman_bot.log", encoding='utf-8')
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)

# Create formatters and add them to handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# =======================
# Timezone Configuration
# =======================

TIMEZONE = pytz.timezone(config['General']['timezone'])

# =======================
# OpenAI Client Setup
# =======================

client = OpenAI(api_key=config['OpenAI']['api_key'])

# OpenAI configuration
model = config['OpenAI'].get('model', 'gpt-4')

# =======================
# Email Configuration
# =======================

# SMTP & IMAP configuration
smtp_server = config['Email']['smtp_server']
smtp_port = int(config['Email']['smtp_port'])
smtp_email = config['Email']['email']
smtp_password = config['Email']['password']

imap_server = config['Email']['imap_server']
imap_email = config['Email']['email']
imap_password = config['Email']['password']

imap_check_interval = int(config['Email']['imap_check_interval'])

# =======================
# SMS Configuration
# =======================

# SMS recipient configuration
# recipient_carrier_gateway = config['SMS']['recipient_carrier_gateway']
# recipient_number = config['SMS']['recipient_number']
# TEST_USER_NUMBER = config['SMS']['recipient_number']
TEST_USER_NUMBER = None

# =======================
# MongoDB Setup
# =======================

mongo_client = AsyncIOMotorClient(
    host=config['MongoDB']['host'],
    port=int(config['MongoDB']['port']),
    username=config['MongoDB']['username'],
    password=config['MongoDB']['password'],
    directConnection=True
)
mongo_db = mongo_client[config['MongoDB']['db']]
users_collection = mongo_db[config['MongoDB']['user_collection']]
prompt_collection = mongo_db[config['MongoDB']['prompt_collection']]

# =======================
# Data Models
# =======================

class UserPreferences:
    def __init__(self, phone_number: str, carrier_gateway: str, _id=None, **kwargs):
        self.phone_number = phone_number
        self.carrier_gateway = carrier_gateway
        self.personal_info: Dict = kwargs.get('personal_info', {})
        self.conversation: List[Dict] = kwargs.get('conversation', [])
        self.created_at: datetime = kwargs.get('created_at', datetime.now(timezone.utc))
        self.updated_at: datetime = kwargs.get('updated_at', datetime.now(timezone.utc))
        self.status: str = kwargs.get('status', "pending")  # "pending", "active", "declined", "awaiting_character_selection", "active_with_character"
        self.bot_name: Optional[str] = kwargs.get('bot_name')  # To store selected character's name
        self.timezone: str = kwargs.get('timezone', "America/Chicago")  # Default to America/Chicago if not set

    def to_dict(self):
        return {
            "phone_number": self.phone_number,
            "carrier_gateway": self.carrier_gateway,
            "personal_info": self.personal_info,
            "conversation": self.conversation,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "bot_name": self.bot_name,
            "timezone": self.timezone
        }

class Prompts:
    def __init__(self, _id=None, **kwargs):
        self.system_prompt: str = kwargs.get(
            'system_prompt',
            "You are a charismatic boyfriend who sends compliments and encouraging text messages to his girlfriend. "
            "You are energetic, fun-loving, easy-going, and act like a gentleman while also being a bit playful and flirtatious. "
            "Messages should be informal, casual, and no more than 240 characters in length."
        )
        self.morning_prompt: str = kwargs.get(
            'morning_prompt',
            "Write a good morning text message to a girlfriend that is complimentary, motivating, and demonstrates your caring nature."
        )
        self.afternoon_prompt: str = kwargs.get(
            'afternoon_prompt',
            "Write a good afternoon text message to a girlfriend that is thoughtful, engaging, and keeps the conversation lively."
        )
        self.evening_prompt: str = kwargs.get(
            'evening_prompt',
            "Write a good evening text message to a girlfriend that is relaxing, affectionate, and wraps up the day positively."
        )

    def to_dict(self):
        return {
            "system_prompt": self.system_prompt,
            "morning_prompt": self.morning_prompt,
            "afternoon_prompt": self.afternoon_prompt,
            "evening_prompt": self.evening_prompt
        }


# =======================
# Database Operations
# =======================

async def get_user(phone_number: str) -> Optional[UserPreferences]:
    logger.debug(f"Attempting to retrieve user with phone_number={phone_number}")
    user_data = await users_collection.find_one({"phone_number": phone_number}, {'_id': 0})
    if user_data:
        logger.debug(f"User found for {phone_number}: {json.dumps(user_data, default=str)}")
        return UserPreferences(**user_data)
    logger.debug(f"No user found for {phone_number}")
    return None

async def create_user(phone_number: str, carrier_gateway: str, status="pending") -> UserPreferences:
    logger.info(f"Creating new user with phone_number={phone_number} and status={status}")
    
    # Determine the timezone based on the phone number
    user_timezone = get_timezone_from_phone_number(phone_number)
    
    user = UserPreferences(phone_number, carrier_gateway, status=status, timezone=user_timezone)
    await users_collection.insert_one(user.to_dict())
    logger.debug(f"User created: {json.dumps(user.to_dict(), default=str)}")
    return user

async def update_user_preferences(phone_number: str, updates: Dict) -> None:
    updates["updated_at"] = datetime.now(timezone.utc)
    logger.debug(f"Updating user {phone_number} with updates: {updates}")
    await users_collection.update_one(
        {"phone_number": phone_number},
        {"$set": updates}
    )
    logger.info(f"User preferences updated for {phone_number}")

async def get_prompts() -> Optional[Prompts]:
    logger.debug("Retrieving system prompts from the database")
    prompt_data = await prompt_collection.find_one({}, {'_id': 0})
    if prompt_data:
        logger.debug(f"Prompts found: {json.dumps(prompt_data, default=str)}")
        return Prompts(**prompt_data)
    logger.warning("No system prompts found in the database")
    return None

# =======================
# Define Characters
# =======================

CHARACTERS = [
    {
        "name": "Zach",
        "image_url": "https://assets.rndmzd.com/zach.png"
    },
    {
        "name": "Tyler",
        "image_url": "https://assets.rndmzd.com/tyler.png"
    },
    {
        "name": "Jake",
        "image_url": "https://assets.rndmzd.com/jake.png"
    },
    {
        "name": "Chase",
        "image_url": "https://assets.rndmzd.com/chase.png"
    }
]

# =======================
# OpenAI Interaction
# =======================

async def generate_message(incoming_message: str, phone_number: str):
    logger.info(f"Generating message for incoming_message='{incoming_message}' from phone_number={phone_number}")
    user = await get_user(phone_number)
    if not user:
        logger.warning(f"No user found for {phone_number} during generate_message call.")
        return "You need to confirm first."

    # Retrieve system prompts
    prompts = await get_prompts()
    if not prompts:
        logger.warning("No system prompts found, using default prompts.")
        prompts = Prompts()

    # Prepare conversation for OpenAI
    if not user.conversation:
        # Initialize conversation with system prompt
        conversation = [
            {"role": "system", "content": prompts.system_prompt}
        ]
    else:
        conversation = user.conversation.copy()

    # Append the new user message
    conversation.append({"role": "user", "content": incoming_message})

    # Append bot_name if it exists to personalize the system prompt
    if user.bot_name:
        # Optionally, adjust the system prompt or include a message to set the bot's name
        # Here, we assume the system prompt already uses the bot's name
        pass

    # Truncate conversation if it's too long (optional, based on token limits)
    MAX_CONVERSATION_LENGTH = 20  # Adjust as needed
    if len(conversation) > MAX_CONVERSATION_LENGTH:
        conversation = conversation[-MAX_CONVERSATION_LENGTH:]

    logger.debug(f"Sending request to OpenAI with conversation: {json.dumps(conversation, default=str)}")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=conversation,
            max_tokens=500,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6
        )
        result = response.choices[0].message.content.strip()
        logger.debug(f"OpenAI response: {result}")
        return result
    except Exception as e:
        logger.exception(f"Error generating message from OpenAI: {e}")
        return "Sorry, I'm having trouble processing your request right now."

async def extract_name_from_message(message: str) -> Optional[str]:
    """Use OpenAI to extract a name from a message if present."""
    logger.debug(f"Attempting to extract name from message: {message}")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a name extractor. If there is a name in the message, respond with ONLY the name. If no name is found, respond with 'NONE'. Example: 'Hi this is John' -> 'John', 'Hello there' -> 'NONE'"},
                {"role": "user", "content": message}
            ],
            max_tokens=50
        )
        
        extracted_name = response.choices[0].message.content.strip()
        logger.debug(f"Extracted name: {extracted_name}")
        
        return None if extracted_name.upper() == "NONE" else extracted_name
    except Exception as e:
        logger.error(f"Error extracting name: {e}")
        return None

async def extract_character_selection(message: str) -> Optional[str]:
    """Use OpenAI to extract character selection from the user's message."""
    logger.debug(f"Attempting to extract character selection from message: {message}")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a character selection assistant. Extract the character's name from the user's reply. If the user replies with a number (1-4), map it to the corresponding character name. Respond with ONLY the character's name. If no valid selection is found, respond with 'INVALID'."},
                {"role": "user", "content": message}
            ],
            max_tokens=50
        )
        
        extracted_selection = response.choices[0].message.content.strip()
        logger.debug(f"Extracted character selection: {extracted_selection}")
        
        return None if extracted_selection.upper() == "INVALID" else extracted_selection
    except Exception as e:
        logger.error(f"Error extracting character selection: {e}")
        return None

# =======================
# SMS Sending Function
# =======================

async def send_sms(message: str, recipient_number: str, recipient_carrier_gateway: Optional[str] = None):
    if not recipient_carrier_gateway:
        user = await get_user(recipient_number)
        if not user:
            logger.warning(f"No user found for {recipient_number} during send_sms call.")
            return
        recipient_carrier_gateway = user.carrier_gateway
        if not recipient_carrier_gateway:
            logger.warning(f"No carrier gateway found for {recipient_number} during send_sms call.")
            return
    logger.info(f"Sending SMS to {recipient_number}@{recipient_carrier_gateway} with message: {message}")
    recipient_email = f"{recipient_number}@{recipient_carrier_gateway}"
    msg = MIMEText(message)
    msg['From'] = smtp_email
    msg['To'] = recipient_email
    # msg['Subject'] = "Gentleman Bot"  # Optional: Depending on carrier gateway requirements

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_email, smtp_password)
            server.sendmail(smtp_email, recipient_email, msg.as_string())
        logger.info(f"SMS sent successfully to {recipient_email}")
    except Exception as e:
        logger.exception(f"Error sending SMS to {recipient_email}: {e}")

# =======================
# Daily Cron Job
# =======================

@aiocron.crontab('* * * * *')  # Runs every minute
async def daily_message():
    logger.info("Running daily_message cron job...")
    users = await get_all_active_users()
    if not users:
        logger.info("No active users found, skipping daily message.")
        return

    prompts = await get_prompts()
    if not prompts:
        logger.warning("No prompts found, skipping daily message.")
        return

    for user in users:
        if not user.timezone:
            logger.warning(f"User {user.phone_number} does not have a timezone set. Skipping message.")
            continue

        try:
            user_timezone = pytz.timezone(user.timezone)
        except pytz.UnknownTimeZoneError:
            logger.error(f"Unknown timezone '{user.timezone}' for user {user.phone_number}. Skipping message.")
            continue

        # Get current time in user's timezone
        current_time = datetime.now(user_timezone)
        current_hour = current_time.hour

        # Determine if it's time to send a message
        # For example, send at 9 AM, 3 PM, and 9 PM
        if current_hour in [9, 15, 21] and current_time.minute == 0:
            if 5 <= current_hour < 12:
                prompt = prompts.morning_prompt
                time_of_day = "morning"
            elif 12 <= current_hour < 17:
                prompt = prompts.afternoon_prompt
                time_of_day = "afternoon"
            elif 17 <= current_hour < 22:
                prompt = prompts.evening_prompt
                time_of_day = "evening"
            else:
                logger.info(f"User {user.phone_number}: Current time {current_hour}:00 is outside defined sending hours.")
                continue

            # Generate customized message
            message = await generate_message(prompt, user.phone_number)
            if message:
                await send_sms(message, user.phone_number)
                logger.info(f"Daily {time_of_day} message sent successfully to {user.phone_number}.")
            else:
                logger.warning(f"No message generated for user {user.phone_number} at {time_of_day}.")

async def get_all_active_users() -> List[UserPreferences]:
    logger.debug("Retrieving all active users with selected characters.")
    users = []
    cursor = users_collection.find({"status": "active_with_character"})
    async for user_data in cursor:
        users.append(UserPreferences(**user_data))
    logger.debug(f"Retrieved {len(users)} active users.")
    return users



# =======================
# IMAP Email Processing
# =======================

def check_imap_inbox() -> List[Dict]:
    """
    Check the IMAP inbox for UNSEEN emails and return a list of messages.
    Each message is a dict with 'msg_id', 'from', and 'body'.
    """
    logger.debug("Connecting to IMAP server to check for new messages.")
    try:
        mail = imaplib.IMAP4_SSL(imap_server)
        mail.login(imap_email, imap_password)
        mail.select("INBOX")  # You can specify a different folder if needed

        logger.debug("Searching for UNSEEN messages.")
        status, msg_ids = mail.search(None, '(UNSEEN)')
        messages = []
        if status == 'OK':
            msg_id_list = msg_ids[0].split()
            logger.debug(f"UNSEEN message IDs returned: {msg_id_list}")
            for msg_id in msg_id_list:
                logger.debug(f"Fetching message with ID {msg_id}")
                status, msg_data = mail.fetch(msg_id, '(RFC822)')
                if status != 'OK':
                    logger.warning(f"Failed to fetch message with ID {msg_id}")
                    continue
                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)

                from_ = msg.get("From", "")
                subject = msg.get("Subject", "")

                # Decode the subject if necessary
                decoded_subject = ""
                for part, enc in decode_header(subject):
                    if isinstance(part, bytes):
                        decoded_subject += part.decode(enc or 'utf-8', errors='replace')
                    else:
                        decoded_subject += part

                # Extract the body
                body = None
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain" and not part.get('Content-Disposition'):
                            payload = part.get_payload(decode=True)
                            if payload:
                                charset = part.get_content_charset() or 'utf-8'
                                body = payload.decode(charset, errors='replace')
                                break
                else:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        charset = msg.get_content_charset() or 'utf-8'
                        body = payload.decode(charset, errors='replace')

                if body:
                    messages.append({
                        "msg_id": msg_id.decode(),
                        "from": from_,
                        "body": body.strip()
                    })
                    logger.debug(f"Extracted body for msg_id={msg_id.decode()}: {body.strip()}")
                else:
                    logger.debug(f"No body found for message {msg_id.decode()}")

        else:
            logger.warning("Failed to search for UNSEEN messages in IMAP.")

        mail.close()
        mail.logout()
        logger.debug(f"Found {len(messages)} new messages.")
        return messages

    except Exception as e:
        logger.exception(f"Error checking IMAP inbox: {e}")
        return []

def mark_email_as_read(msg_id: str):
    """
    Mark an email as read/seen in IMAP.
    """
    logger.debug(f"Marking email with ID {msg_id} as read.")
    try:
        mail = imaplib.IMAP4_SSL(imap_server)
        mail.login(imap_email, imap_password)
        mail.select("INBOX")
        mail.store(msg_id, '+FLAGS', '\\Seen')
        mail.close()
        mail.logout()
        logger.debug(f"Email with ID {msg_id} marked as read.")
    except Exception as e:
        logger.exception(f"Error marking email {msg_id} as read: {e}")

async def process_inbound_emails():
    """
    Process inbound emails by checking the IMAP inbox and handling each message accordingly.
    """
    logger.debug("Starting process to handle inbound emails.")
    messages = await asyncio.to_thread(check_imap_inbox)
    if not messages:
        logger.debug("No new emails to process.")
        return

    for message in messages:
        msg_id = message['msg_id']
        from_email = message['from']
        body = message['body']
        logger.debug(f"Processing email with msg_id={msg_id}, from={from_email}")

        # Extract phone number from the From address
        # Assumption: sender email is in the format "1234567890@carrier_gateway.com"
        sender_part = from_email.split('@')[0][-10:] if '@' in from_email else from_email[-10:]
        logger.debug(f"Extracted phone number from email: {sender_part}")

        # Retrieve user
        user = await get_user(sender_part)
        logger.debug(f"Retrieved user from DB: {user.to_dict() if user else None}")

        if user is None:
            carrier_gateway = from_email.split('@')[1] if '@' in from_email else ""
            logger.debug(f"Carrier_gateway extracted from email: {carrier_gateway}")
            # New user: create with status 'pending' and ask for confirmation
            user = await create_user(sender_part, carrier_gateway, status="pending")
            confirmation_message = "Hi! You are about to connect with your perfect gentleman. Reply 'Yes' to continue or 'No' to decline."
            await send_sms(confirmation_message, recipient_number=sender_part)
            logger.info(f"Sent confirmation message to new user {sender_part}")
            mark_email_as_read(msg_id)
            continue

        # Send character selection prompt with images (as URLs)
        character_selection_message = (
            f"Great! Please select one of the following characters by replying with the corresponding number:\n"
            + "\n".join([f"{i+1}. {char['name']}\n{char['image_url']}" for i, char in enumerate(CHARACTERS)])
        )

        if user.status == "pending":
            # Expecting "Yes" or "No"
            if body.lower() == "yes":
                await update_user_preferences(sender_part, {"status": "awaiting_character_selection"})
                await send_sms(character_selection_message, recipient_number=sender_part)
                logger.info(f"Sent character selection message to user {sender_part}")
            elif body.lower() == "no":
                await update_user_preferences(sender_part, {"status": "declined"})
                goodbye_message = "No problem. Have a great day!"
                await send_sms(goodbye_message, recipient_number=sender_part)
                logger.info(f"User {sender_part} declined to activate.")
            else:
                # Invalid response, remind the user
                reminder_message = "Please reply with 'Yes' or 'No' to continue."
                await send_sms(reminder_message, recipient_number=sender_part)
                logger.info(f"User {sender_part} provided invalid confirmation response.")
            mark_email_as_read(msg_id)
            continue

        if user.status == "awaiting_character_selection":
            # Expecting a number between 1 and 4
            try:
                selection = int(body.strip())
                if 1 <= selection <= 4:
                    selected_character = CHARACTERS[selection - 1]
                    await update_user_preferences(sender_part, {
                        "status": "active_with_character",
                        "bot_name": selected_character["name"],
                        "personal_info": {"character_name": selected_character["name"]}
                    })
                    selection_ack_message = f"Awesome! I'm {selected_character['name']}, and I'm here to be your perfect gentleman. What's your name?"
                    await send_sms(selection_ack_message, recipient_number=sender_part)
                    logger.info(f"User {sender_part} selected character {selected_character['name']}")
                    
                    # Update conversation with a system message to set the bot's name
                    user.conversation.append({"role": "system", "content": f"You are now {selected_character['name']}, a charismatic boyfriend who sends compliments and encouraging text messages to his girlfriend. You are energetic, fun-loving, easy-going, and always act like a gentleman. Messages should be informal, casual, and no more than 240 characters in length."})
                    await update_user_preferences(sender_part, {"conversation": user.conversation})
                else:
                    # Invalid selection
                    invalid_selection_message = "Invalid selection. Please reply with a number between 1 and 4 to select your perfect gentleman."
                    await send_sms(invalid_selection_message, recipient_number=sender_part)
                    logger.info(f"User {sender_part} provided an invalid selection number: {selection}")
            except ValueError:
                # Non-integer response
                invalid_selection_message = "Invalid input. Please reply with a number between 1 and 4 to select your perfect gentleman."
                await send_sms(invalid_selection_message, recipient_number=sender_part)
                logger.info(f"User {sender_part} provided a non-integer selection: {body}")
            mark_email_as_read(msg_id)
            continue

        if user.status == "declined":
            # User has previously declined; you can choose to ignore or allow them to re-enable
            # Here, we allow re-enabling by replying 'Yes'
            if body.lower() == "yes":
                await update_user_preferences(sender_part, {"status": "awaiting_character_selection"})
                await send_sms(character_selection_message, recipient_number=sender_part)
                logger.info(f"Sent character selection message to reactivated user {sender_part}")
            else:
                # Optionally ignore or send a polite reminder
                ignore_message = "You previously declined. If you change your mind, just reply 'Yes'."
                await send_sms(ignore_message, recipient_number=sender_part)
                logger.info(f"User {sender_part} sent a message after declining; sent reminder.")
            mark_email_as_read(msg_id)
            continue

        if user.status == "active_with_character":
            logger.info(f"Processing message from active user {sender_part}: {body}")

            # Check if the message is "stop" (case-insensitive and stripped)
            if body.lower().strip() == "stop":
                await update_user_preferences(sender_part, {"status": "declined"})
                stop_confirmation_message = (
                    "You have been unsubscribed and will no longer receive messages from your perfect gentleman. "
                    "If you wish to resume, reply with 'Yes' at any time."
                )
                await send_sms(stop_confirmation_message, recipient_number=sender_part)
                logger.info(f"User {sender_part} has unsubscribed from receiving messages.")
                mark_email_as_read(msg_id)
                continue

            # If the message is not "stop", proceed normally
            # Optionally, extract the user's name if not already done
            if not user.personal_info.get("name"):
                # Extract name from message
                extracted_name = await extract_name_from_message(body)
                if extracted_name:
                    updated_info = user.personal_info.copy()
                    updated_info["name"] = extracted_name
                    await update_user_preferences(sender_part, {"personal_info": updated_info})
                    logger.info(f"Updated personal_info in DB for {sender_part}: {updated_info}")
                else:
                    logger.info(f"No name extracted from message for {sender_part}")

            # Append user message to conversation
            user.conversation.append({"role": "user", "content": body})

            # Generate response with conversation context
            response_message = await generate_message(body, sender_part)
            if response_message:
                # Append assistant's reply to conversation
                user.conversation.append({"role": "assistant", "content": response_message})
                await send_sms(response_message, recipient_number=sender_part)
                logger.info(f"Sent response to {sender_part}: {response_message}")
                # Update conversation in DB
                await update_user_preferences(sender_part, {"conversation": user.conversation})
            else:
                logger.warning(f"No response generated for message from {sender_part}")

            # Mark email as read
            mark_email_as_read(msg_id)
            continue

def get_timezone_from_phone_number(phone_number: str) -> str:
    """
    Parses the phone number and returns the primary timezone based on the area code.
    If unable to determine, defaults to 'America/Chicago'.
    """
    try:
        # Parse the phone number. Assume it's a US number.
        parsed_number = phonenumbers.parse(phone_number, "US")
        
        # Get the timezone(s) for the number
        time_zones = ph_timezone.time_zones_for_number(parsed_number)
        
        if time_zones:
            # Return the first timezone in the list
            return time_zones[0]
        else:
            logger.warning(f"No timezone found for phone number: {phone_number}. Defaulting to America/Chicago.")
            return "America/Chicago"
    except phonenumbers.NumberParseException as e:
        logger.error(f"Error parsing phone number {phone_number}: {e}")
        return "America/Chicago"


# =======================
# Main Application
# =======================

async def main():
    logger.info("Gentleman Bot is starting up...")

    # Start periodic IMAP checking
    async def imap_check_loop():
        logger.debug("Starting IMAP check loop...")
        while True:
            await process_inbound_emails()
            await asyncio.sleep(imap_check_interval)  # Check every 60 seconds

    asyncio.create_task(imap_check_loop())

    # Optionally send an initial message to a known recipient_number if active_with_character
    if TEST_USER_NUMBER:
        user = await get_user(TEST_USER_NUMBER)
        if user and user.status == "active_with_character":
            prompts = await get_prompts()
            if prompts:
                default_message = prompts.morning_prompt
                logger.info(f"Initial default_message: {default_message}")
                response_message = await generate_message(default_message, TEST_USER_NUMBER)
                if response_message:
                    # Append assistant's reply to conversation
                    user.conversation.append({"role": "assistant", "content": response_message})
                    await send_sms(response_message, recipient_number=TEST_USER_NUMBER)
                    # Update conversation in DB
                    await update_user_preferences(TEST_USER_NUMBER, {"conversation": user.conversation})
                    logger.info("Initial daily message sent.")
                else:
                    logger.warning("No response generated for initial daily message.")
            else:
                logger.warning("No prompts found, skipping initial daily message.")
        else:
            logger.info(f"No active user found for {TEST_USER_NUMBER}, skipping initial daily message.")

    try:
        while True:
            logger.debug("Main loop sleeping for 1 hour.")
            await asyncio.sleep(3600)  # Sleep for 1 hour
    except asyncio.CancelledError:
        logger.info("Shutting down gracefully...")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down gracefully...")
