from tools.email_tool import email_sender_tool
import os
from dotenv import load_dotenv

load_dotenv()

def test_email():
    to_email = os.getenv("EMAIL_ADDRESS") # Send to self for testing
    if not to_email:
        print("EMAIL_ADDRESS not found in .env")
        return
        
    subject = "Test Email from AI Recruitment Agent"
    body = "This is a test email to verify SMTP configuration."
    
    print(f"Attempting to send email to {to_email}...")
    result = email_sender_tool.invoke({"to_email": to_email, "subject": subject, "body": body})
    print(result)

if __name__ == "__main__":
    test_email()
