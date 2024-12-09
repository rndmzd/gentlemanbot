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
# import re
import json

# =======================
# Configuration and Logging
# =======================

config = configparser.ConfigParser()
config.read('config.ini')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gentleman_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

# =======================
# SMS Configuration
# =======================

# SMS recipient configuration
# recipient_carrier_gateway = config['SMS']['recipient_carrier_gateway']
# recipient_number = config['SMS']['recipient_number']
TEST_USER_NUMBER = config['SMS']['recipient_number']

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
        self.status: str = kwargs.get('status', "pending")  # "pending", "active", "declined"

    def to_dict(self):
        return {
            "phone_number": self.phone_number,
            "carrier_gateway": self.carrier_gateway,
            "personal_info": self.personal_info,
            "conversation": self.conversation,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status
        }

class Prompts:
    def __init__(self, _id=None, **kwargs):
        self.system_prompt: str = kwargs.get(
            'system_prompt',
            "You are a charismatic boyfriend who sends compliments and encouraging text messages to his girlfriend. "
            "You are energetic, fun-loving, easy-going, and always act like a gentleman. "
            "Messages should be informal, casual, and no more than 240 characters in length."
        )
        self.morning_prompt: str = kwargs.get(
            'morning_prompt',
            "Write a morning text message to a girlfriend that is complimentary, motivating, and demonstrates your caring nature."
        )

    def to_dict(self):
        return {
            "system_prompt": self.system_prompt,
            "morning_prompt": self.morning_prompt
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
    user = UserPreferences(phone_number, carrier_gateway, status=status)
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
    conversation = prompts.conversation.copy() if hasattr(prompts, 'conversation') else []

    # If conversation history exists, use it; else, initialize with system prompt
    if not user.conversation:
        conversation = [
            {"role": "system", "content": prompts.system_prompt}
        ]
    else:
        conversation = user.conversation.copy()

    # Append the new user message
    conversation.append({"role": "user", "content": incoming_message})

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
        
        return None if extracted_name == "NONE" else extracted_name
    except Exception as e:
        logger.error(f"Error extracting name: {e}")
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

@aiocron.crontab('0 8 * * *')  # Runs at 8:00 AM every day
async def daily_message(recipient_number: str):
    logger.info("Running daily_message cron job...")
    user = await get_user(recipient_number)
    if not user or user.status != "active":
        logger.info("User is not active or doesn't exist, skipping daily message.")
        return

    prompts = await get_prompts()
    if not prompts:
        logger.warning("No prompts found, skipping daily message.")
        return

    # Generate daily message
    message = await generate_message(prompts.morning_prompt, recipient_number)
    if message:
        await send_sms(message)
        logger.info("Daily message sent successfully.")
    else:
        logger.warning("No message generated for daily message.")

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

        if user is None:
            carrier_gateway = from_email.split('@')[1] if '@' in from_email else ""
            logger.debug(f"Carrier_gateway extracted from email: {carrier_gateway}")
            # New user: create with status 'pending' and ask for confirmation
            user = await create_user(sender_part, carrier_gateway, status="pending")
            confirmation_message = "Hi! You are about to connect with Gentleman Bot. Reply 'Yes' to continue or 'No' to decline."
            await send_sms(confirmation_message, recipient_number=sender_part)
            logger.info(f"Sent confirmation message to new user {sender_part}")
            mark_email_as_read(msg_id)
            continue

        if user.status == "pending":
            # Expecting "Yes" or "No"
            if body.lower() == "yes":
                await update_user_preferences(sender_part, {"status": "active"})
                # welcome_message = "Great! You are now connected with Gentleman Bot. How can I help you today?"
                welcome_message = "Hey there, beautiful! I'm so glad you wanted to connect. What's your name?"
                await send_sms(welcome_message, recipient_number=sender_part)
                logger.info(f"User {sender_part} confirmed to activate.")
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

        if user.status == "declined":
            # User has previously declined; you can choose to ignore or allow them to re-enable
            # Here, we allow re-enabling by replying 'Yes'
            if body.lower() == "yes":
                await update_user_preferences(sender_part, {"status": "active"})
                reactivation_message = "Welcome back! You are now connected with Gentleman Bot. How can I assist you today?"
                await send_sms(reactivation_message, recipient_number=sender_part)
                logger.info(f"User {sender_part} reactivated.")
            else:
                # Optionally ignore or send a polite reminder
                ignore_message = "You previously declined. If you change your mind, just reply 'Yes'."
                await send_sms(ignore_message, recipient_number=sender_part)
                logger.info(f"User {sender_part} sent a message after declining; sent reminder.")
            mark_email_as_read(msg_id)
            continue

        if user.status == "active":
            logger.info(f"Processing message from active user {sender_part}: {body}")

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
            await asyncio.sleep(60)  # Check every 60 seconds

    asyncio.create_task(imap_check_loop())

    # Optionally send an initial message to a known recipient_number if active
    user = await get_user(TEST_USER_NUMBER)
    if user and user.status == "active":
        prompts = await get_prompts()
        if prompts:
            default_message = prompts.morning_prompt
            logger.info(f"Initial default_message: {default_message}")
            response_message = await generate_message(default_message, TEST_USER_NUMBER)
            if response_message:
                # Append assistant's reply to conversation
                user.conversation.append({"role": "assistant", "content": response_message})
                await send_sms(response_message)
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
