import json
from datetime import datetime
import speech_recognition as sr
from langdetect import detect
from deep_translator import GoogleTranslator
from supabase import create_client, Client

# Supabase configuration
SUPABASE_URL = "https://bliqlgvbgwfedjqpghcn.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJsaXFsZ3ZiZ3dmZWRqcXBnaGNuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDIyMDYzMDAsImV4cCI6MjA1Nzc4MjMwMH0.bFRoPx5PIgEQg-L-UxFUs12H7bMd7TkTiAAYBSm03U8"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Replace with your ngrok URL
NGROK_URL = "https://881a-2402-8100-313a-f6a1-4546-fe32-fd60-608.ngrok-free.app"  # Update this with your actual ngrok URL

def audio_to_text():
    """
    Captures live audio and converts it into text using Google's Speech API.
    """
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Listening... Please speak your banking query.")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)  # Set a timeout for mobile devices

        text = recognizer.recognize_google(audio)
        print(f"Recognized Speech: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print(f"Speech Recognition Error: {e}")
        return ""
    except sr.WaitTimeoutError:
        print("Microphone access timed out. Please ensure microphone permissions are granted.")
        return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

def translate_to_english(text):
    """
    Detects the language and translates it to English if it's not already in English.
    """
    if not text:
        return ""

    try:
        detected_lang = detect(text)
        if detected_lang != "en":
            translated_text = GoogleTranslator(source=detected_lang, target="en").translate(text)
            print(f"Translated Text: {translated_text}")
            return translated_text
        return text
    except Exception as e:
        print(f"Translation Error: {e}")
        return text

def analyze_intent(text):
    """
    Determines banking intent using keyword matching.
    Possible intents: Account, Loan, Fixed Deposit, Demat.
    """
    if not text:
        return []

    text = text.lower()

    # Keywords for banking-related intents
    intents = {
        "account": [
            "account", "khata", "khate", "bank khata", "bankecha khata", "bank khati",
            "bachat khata", "chalu khata", "checking account", "saving account",
            "khata ughdaych aahe", "khata ughda", "bank khata ughdaych aahe",
            "bank khata ughda", "khata band karaycha aahe", "khata band", "bank statement",
            "maza khata kuthe aahe", "majhya khatyat kiti paise aahet", "balance check",
            "account balance", "khate tapashil", "bank khata kramank", "account number",
            "mini statement", "passbook update", "passbook", "khatyat jama", "rakkam transfer",
            "bank transfer", "khata update", "bank khatyacha prakaar"
        ],
        "loan": [
            "loan", "karj", "home loan", "gruha karj", "makan karj", "gharasathi karj",
            "car loan", "vahan karj", "two wheeler loan", "char chaki loan", "vyavsayik karj",
            "business loan", "shaikshanik karj", "education loan", "vidyarthi karj", "loan status",
            "karjachi mahiti", "karj ghyaych aahe", "karj hapta", "loan EMI",
            "loan repayment", "karj paratfedt", "loan manjur", "loan approval",
            "karj arj", "loan application", "loan eligibility", "karj patrata",
            "vyaj dar", "loan interest rate", "karjache prakaar", "personal loan",
            "gold loan", "sonyache karj", "auto loan", "bankeche karj"
        ],
        "fixed deposit": [
            "fixed deposit", "FD", "FD account", "FD ughdaychi aahe",
            "FD open", "FD band", "FD band karaychi aahe", "fixed deposit",
            "FD vyaj dar", "FD maturity", "FD investment", "invest in FD",
            "bankeche FD", "fixed deposit investment", "FD status",
            "FD paratfedt", "FD hapta", "FD dar", "mudat thev", "term deposit",
            "FD calculator", "bank FD", "FD kasa karaycha", "FD salla",
            "FD prakaar", "laghu mudat thev", "madhya mudat thev", "dirgha mudat thev"
        ],
        "demat": [
            "demat", "demat account", "demat khata", "share market",
            "share trading", "demat khata ughdaycha aahe", "demat account open",
            "shares kharedi", "shares vikri", "mutual fund", "demat kasa ughdaycha",
            "demat charges", "demat fee", "stock investment", "shares kuthe kharedi karayche",
            "SEBI", "stock trading", "demat account ughadne", "share kharedi vikri",
            "demat account kasa chalvaycha", "demat account fayde", "demat ani trading account",
            "demat account transfer", "demat login", "share bazar guntavanuk", "demat account portal",
            "demat account fayde", "DP charges", "demat account opening process",
            "IPO investment", "demat eligibility", "demat online apply"
        ]
    }

    detected_intents = set()

    # Check if specific banking keywords are present
    for intent, keywords in intents.items():
        for keyword in keywords:
            if keyword in text:
                detected_intents.add(intent.capitalize())

    if detected_intents:
        return list(detected_intents)  # Return all matching intents as a list
    return []

from datetime import datetime


def generate_ticket_number(intent, account_number):
    """
    Generates a ticket number in the format: keyword#account_number.
    Example: Loan#123456, Demat#789012
    """
    return f"{intent}#{account_number}"


def store_ticket_in_supabase(ticket_number, account_number, intent):
    """
    Stores the ticket number, account number, and intent in the Supabase table.
    Checks for duplicate ticket numbers before insertion.
    """
    try:
        # Check if the ticket number already exists
        response = supabase.table('service_tickets').select("*").eq('ticket_number', ticket_number).execute()

        if response.data:
            # Ticket number already exists
            error_message = f"Ticket number {ticket_number} already exists."
            print(error_message)
            return False, error_message

        # Insert the new ticket
        response = supabase.table('service_tickets').insert({
            "ticket_number": ticket_number,
            "account_number": account_number,
            "user_query": intent,
            "query_date": datetime.now().isoformat(),
            "status":"Active"
        }).execute()

        if response.data:
            print("Ticket stored successfully in Supabase.")
            return True, None  # Success
        else:
            error_message = "Failed to store ticket in Supabase."
            print(error_message)
            return False, error_message
    except Exception as e:
        error_message = f"Supabase Error: {e}"
        print(error_message)
        return False, error_message


def process_audio(account_number):
    """
    Full pipeline: Capture audio -> Convert to text -> Translate if needed -> Analyze intent -> Generate ticket -> Return JSON.
    Handles duplicate ticket numbers by returning an alert message.
    """
    text = audio_to_text()
    if text:
        translated_text = translate_to_english(text)
        intent = analyze_intent(translated_text)
        print(f"Final Detected Intent: {intent}")

        if intent:
            # Generate a ticket number in the format: keyword#account_number
            ticket_number = generate_ticket_number(intent[0], account_number)

            # Store the ticket in Supabase
            success, error_message = store_ticket_in_supabase(ticket_number, account_number, intent[0])
            if not success:
                # Handle duplicate ticket case
                if "already exists" in error_message:
                    return json.dumps({"status": "error", "message": "This ticket already exists. Please try again."})
                else:
                    return json.dumps({"status": "error", "message": error_message})

            # Return the ticket number and other details as JSON
            return json.dumps({
                "status": "success",
                "message": "Ticket generated successfully.",
                "ticket_number": ticket_number,
                "account_number": account_number,
                "intent": intent[0],
                "redirect_url": f"{NGROK_URL}/{intent[0].lower()}?account_number={account_number}&ticket_number={ticket_number}"
            })
        else:
            return json.dumps({"status": "error", "message": "No valid intent detected."})
    else:
        return json.dumps({"status": "error", "message": "No valid input detected."})


if __name__ == "__main__":
    # Example: Replace this with the actual account number from the dashboard or session
    account_number = "15"  # Replace with dynamic account number
    result = process_audio(account_number)
    print(result)  # Print the JSON response