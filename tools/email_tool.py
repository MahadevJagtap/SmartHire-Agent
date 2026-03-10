import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_core.tools import tool
import os

@tool
def email_sender_tool(to_email: str, subject: str, body: str) -> str:
    """
    Sends an email using SMTP. 
    Requires environment variables: SMTP_SERVER, SMTP_PORT, EMAIL_ADDRESS, EMAIL_PASSWORD.
    """
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("EMAIL_ADDRESS") or os.getenv("SMTP_USER")
    smtp_password = os.getenv("EMAIL_PASSWORD") or os.getenv("SMTP_PASSWORD")
    recruiter_name = os.getenv("RECRUITER_NAME", "AI Recruitment Agent")

    if not smtp_user or not smtp_password:
        return f"SIMULATION: Email sent to {to_email} with subject: {subject}. (SMTP credentials missing)"

    try:
        # Append signature if not already there
        if recruiter_name not in body:
            body += f"\n\nBest regards,\n{recruiter_name}"

        msg = MIMEMultipart()
        msg['From'] = f"{recruiter_name} <{smtp_user}>"
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()
        return f"Success: Email sent to {to_email}"
    except Exception as e:
        return f"Error sending email: {str(e)}"
