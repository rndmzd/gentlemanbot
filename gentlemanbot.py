import configparser
import smtplib
from email.mime.text import MIMEText
from openai import OpenAI
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone
from typing import Optional, Dict
import logging
import asyncio
import aiocron
import imaplib
import email
from email.header import decode_header

config = configparser.ConfigParser()
config.read('config.ini')

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

client = OpenAI(api_key=config['OpenAI']['api_key'])

# OpenAI configuration
model = config['OpenAI'].get('model', 'gpt-4o')

# SMTP & IMAP configuration
smtp_server = config['Email']['smtp_server']
smtp_port = int(config['Email']['smtp_port'])
smtp_email = config['Email']['email']
smtp_password = config['Email']['password']
imap_server = config['Email']['imap_server']
imap_email = config['Email']['email']
imap_password = config['Email']['password']

# SMS recipient configuration
recipient_carrier_gateway = config['SMS']['recipient_carrier_gateway']
recipient_number = config['SMS']['recipient_number']

# MongoDB setup
mongo_client = AsyncIOMotorClient(
    host=config['MongoDB']['host'],
    port=int(config['MongoDB']['port']),
    username=config['MongoDB']['username'],
    password=config['MongoDB']['password'],
    directConnection=True
)
mongo_db = mongo_client[config['MongoDB']['db']]
users_collection = mongo_db[config['MongoDB']['collection']]

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

async def get_or_create_user(phone_number: str) -> UserPreferences:
    logger.debug(f"Attempting to retrieve user with phone_number={phone_number}")
    user_data = await users_collection.find_one(
        {"phone_number": phone_number},
        {'_id': 0}
    )
    if not user_data:
        logger.info(f"No existing user found for {phone_number}. Creating a new user.")
        user = UserPreferences(phone_number)
        await users_collection.insert_one(user.to_dict())
        return user
    logger.debug(f"User found for {phone_number}: {user_data}")
    return UserPreferences(**user_data)

async def update_user_preferences(phone_number: str, updates: Dict) -> None:
    logger.debug(f"Updating user preferences for {phone_number} with updates: {updates}")
    updates["updated_at"] = datetime.now(timezone.utc)
    await users_collection.update_one(
        {"phone_number": phone_number},
        {"$set": updates}
    )
    logger.info(f"User preferences updated for {phone_number}")

async def generate_message(incoming_message: str, phone_number: str):
    logger.info(f"Generating message for incoming_message='{incoming_message}' from phone_number={phone_number}")
    user = await get_or_create_user(phone_number)
    
    context = ""
    if user.personal_info:
        context = "Here's some context about the user: " + str(user.personal_info)
    
    logger.debug(f"Sending request to OpenAI with system_prompt='{user.system_prompt}', context='{context}'")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": user.system_prompt},
            {"role": "system", "content": context},
            {"role": "user", "content": incoming_message}
        ],
        max_tokens=500
    )
    result = response.choices[0].message.content.strip()
    logger.debug(f"OpenAI response: {result}")
    return result

async def send_sms(message):
    logger.info(f"Sending SMS to {recipient_number}@{recipient_carrier_gateway} with message: {message}")
    recipient = f"{recipient_number}@{recipient_carrier_gateway}"
    msg = MIMEText(message)
    msg['From'] = smtp_email
    msg['To'] = recipient

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_email, smtp_password)
            server.sendmail(smtp_email, recipient, msg.as_string())
        logger.info("SMS sent successfully.")
    except Exception as e:
        logger.exception(f"Error sending message: {e}")

@aiocron.crontab('0 8 * * *')
async def daily_message():
    logger.info("Running daily_message cron job...")
    user = await get_or_create_user(recipient_number)
    message = await generate_message(user.morning_prompt, recipient_number)
    await send_sms(message)
    logger.info("Daily message sent.")

def check_imap_inbox():
    """Check the IMAP inbox for UNSEEN emails and return a list of (msg_id, from_, body) tuples."""
    logger.debug("Connecting to IMAP server to check for new messages.")
    mail = imaplib.IMAP4_SSL(imap_server)
    mail.login(imap_email, imap_password)
    mail.select("INBOX")

    logger.debug("Searching for UNSEEN messages.")
    status, msg_ids = mail.search(None, '(UNSEEN)')
    messages = []
    if status == 'OK':
        logger.debug(f"UNSEEN message IDs returned: {msg_ids[0].split() if msg_ids[0] else []}")
        for msg_id in msg_ids[0].split():
            logger.debug(f"Fetching message with ID {msg_id}")
            status, msg_data = mail.fetch(msg_id, '(RFC822)')
            if status != 'OK':
                logger.warning(f"Failed to fetch message with ID {msg_id}")
                continue
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)

            from_ = msg.get("From", "")
            subject = msg.get("Subject", "")

            decoded_subject = ""
            for part, enc in decode_header(subject):
                if isinstance(part, bytes):
                    decoded_subject += part.decode(enc or 'utf-8')
                else:
                    decoded_subject += part

            body = None
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body = payload.decode(part.get_content_charset() or 'utf-8', errors='replace')
                            break
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode(msg.get_content_charset() or 'utf-8', errors='replace')

            if body is not None:
                logger.debug(f"Extracted body for msg_id={msg_id} from={from_}: {body}")
                messages.append((msg_id, from_, body.strip()))
            else:
                logger.debug(f"No body found for message {msg_id}")
    else:
        logger.warning("Failed to search for UNSEEN messages in IMAP.")

    mail.close()
    mail.logout()
    logger.debug(f"Found {len(messages)} new messages.")
    return messages

def mark_email_as_read(msg_id):
    """Mark an email as read/seen in IMAP."""
    logger.debug(f"Marking email with ID {msg_id} as read.")
    mail = imaplib.IMAP4_SSL(imap_server)
    mail.login(imap_email, imap_password)
    mail.select("INBOX")
    mail.store(msg_id, '+FLAGS', '\\Seen')
    mail.close()
    mail.logout()
    logger.debug(f"Email with ID {msg_id} marked as read.")

async def process_inbound_emails():
    logger.debug("Starting process to handle inbound emails.")
    messages = await asyncio.to_thread(check_imap_inbox)
    if not messages:
        logger.debug("No new emails to process.")
    for msg_id, from_, body in messages:
        logger.debug(f"Processing email with msg_id={msg_id}, from={from_}")
        
        # Extract phone number from the From address
        sender_part = from_.split('@')[0][-10:] if '@' in from_ else from_[-10:]
        logger.debug(f'sender_part={sender_part}')

        # Check if it's from the known user
        if sender_part == recipient_number:
            logger.info(f"Processing message from {sender_part} via IMAP: {body}")
            response_message = await generate_message(body, sender_part)
            await send_sms(response_message)
            
            # Mark the email as read
            await asyncio.to_thread(mark_email_as_read, msg_id)
        else:
            logger.debug(f"Email from {from_} [{sender_part}] does not match recipient_number={recipient_number}, ignoring.")

async def main():
    logger.info("Gentleman Bot is starting up...")
    
    user = await get_or_create_user(recipient_number)
    default_message = user.morning_prompt
    logger.info(f"Initial default_message: {default_message}")

    message = await generate_message(default_message, recipient_number)
    logger.info(f"Generated initial message: {message}")

    await send_sms(message)

    async def imap_check_loop():
        logger.debug("Starting IMAP check loop...")
        while True:
            await process_inbound_emails()
            await asyncio.sleep(60)

    asyncio.create_task(imap_check_loop())

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
