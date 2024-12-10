# Gentleman Bot

Gentleman Bot is a personalized SMS chatbot designed to send complimentary, motivating, and engaging text messages to users. Acting as a charismatic boyfriend, Gentleman Bot enhances user experience by delivering thoughtful messages tailored to different times of the day. The bot leverages OpenAI's powerful language models, MongoDB for data storage, and email-to-SMS gateways for message delivery.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Automated Messaging**: Sends daily messages at predefined times (morning, afternoon, evening) customized based on the user's timezone.
- **Subscription Management**: Users can subscribe, unsubscribe, and resubscribe to messages by replying with specific keywords like "yes" or "stop."
- **Personalization**: Extracts user names from messages and adapts conversations accordingly.
- **Timezone Detection**: Automatically determines user timezone based on phone number area codes, ensuring timely message delivery.
- **Character Selection**: Users can choose from a set of predefined characters to personalize their interactions with the bot.
- **Logging**: Comprehensive logging for monitoring and debugging purposes.
- **Scalable Architecture**: Utilizes asynchronous programming with `asyncio` for efficient handling of multiple users.

## Prerequisites

Before setting up Gentleman Bot, ensure you have the following:

- **Python 3.8+**: Ensure Python is installed on your system. You can download it from [here](https://www.python.org/downloads/).
- **MongoDB**: A MongoDB instance for storing user data and prompts. You can set up a local instance or use a cloud service like [MongoDB Atlas](https://www.mongodb.com/cloud/atlas).
- **OpenAI API Key**: Sign up and obtain an API key from [OpenAI](https://platform.openai.com/account/api-keys).
- **Email Account**: An email account (SMTP & IMAP) for sending and receiving SMS via email-to-SMS gateways.
- **Carrier Gateway Information**: Knowledge of your users' carrier gateways to send SMS via email (e.g., `number@carrier.com`).

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/gentleman-bot.git
   cd gentleman-bot
   ```

2. **Create a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *If `requirements.txt` is not provided, install the necessary packages manually:*

   ```bash
   pip install openai motor aiocron phonenumbers pytz
   ```

## Configuration

Gentleman Bot relies on a `config.ini` file for configuration. Follow the steps below to set it up:

1. **Create `config.ini`**

   In the root directory of the project, create a file named `config.ini`.

   ```ini
   [OpenAI]
   api_key = your_openai_api_key
   model = gpt-4

   [Email]
   smtp_server = smtp.your-email-provider.com
   smtp_port = 587
   email = your_email@example.com
   password = your_email_password
   imap_server = imap.your-email-provider.com
   imap_check_interval = 60  # in seconds

   [SMS]
   recipient_number = 1234567890  # Test user number

   [MongoDB]
   host = localhost
   port = 27017
   username = your_mongodb_username
   password = your_mongodb_password
   db = gentleman_bot_db
   user_collection = users
   prompt_collection = prompts
   ```

2. **Configure OpenAI**

   - Replace `your_openai_api_key` with your actual OpenAI API key.
   - Optionally, adjust the `model` if you prefer using a different OpenAI model.

3. **Configure Email**

   - Set `smtp_server` and `smtp_port` according to your email provider's SMTP settings.
   - Replace `your_email@example.com` and `your_email_password` with your email credentials.
   - Set `imap_server` to your email provider's IMAP server.
   - `imap_check_interval` defines how frequently (in seconds) the bot checks for new messages.

4. **Configure SMS**

   - `recipient_number` is used for testing purposes. Replace it with the desired phone number for initial tests.

5. **Configure MongoDB**

   - Set the `host`, `port`, `username`, and `password` according to your MongoDB setup.
   - `db` specifies the database name.
   - `user_collection` and `prompt_collection` define the respective collection names.

## Usage

Once configured, you can start the Gentleman Bot using the following command:

```bash
python gentleman_bot.py
```

*Ensure that the script filename matches the actual Python file. For example, if your main script is named `gentleman_bot.py`, use the command above.*

### Running as a Background Service

For continuous operation, consider running the bot as a background service using tools like `nohup`, `screen`, or `systemd`.

**Using `nohup`:**

```bash
nohup python gentleman_bot.py &
```

**Using `screen`:**

```bash
screen -S gentleman_bot
python gentleman_bot.py
# Press Ctrl+A, then D to detach
```

## Project Structure

```
gentleman-bot/
├── gentleman_bot.py
├── config.ini
├── requirements.txt
├── gentleman_bot.log
├── README.md
└── LICENSE
```

- **gentleman_bot.py**: The main Python script containing the bot's logic.
- **config.ini**: Configuration file for setting up API keys, email credentials, MongoDB details, etc.
- **requirements.txt**: Lists all Python dependencies.
- **gentleman_bot.log**: Log file for monitoring and debugging.
- **README.md**: Project documentation (this file).
- **LICENSE**: License information.

## Contributing

Contributions are welcome! To contribute to Gentleman Bot, follow these steps:

1. **Fork the Repository**

   Click the "Fork" button at the top-right corner of the repository page to create a personal copy.

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/yourusername/gentleman-bot.git
   cd gentleman-bot
   ```

3. **Create a New Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**

   Implement your feature or bug fix.

5. **Commit Your Changes**

   ```bash
   git commit -m "Add feature: description of your feature"
   ```

6. **Push to Your Fork**

   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**

   Navigate to the original repository and click "Compare & pull request." Provide a clear description of your changes.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions, issues, or feature requests, please open an issue on the [GitHub repository](https://github.com/yourusername/gentleman-bot/issues) or contact the maintainer at [your_email@example.com](mailto:your_email@example.com).
