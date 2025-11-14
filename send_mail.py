import smtplib
from email.mime.text import MIMEText

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

USERNAME = "sahilsinglaktr@gmail.com"      # your Gmail
PASSWORD = "jexh xnlw rkkv yhrz"

FROM_EMAIL = USERNAME
TO_EMAIL = "sahil.singla@algo8.ai"
SUBJECT = "Test Email"
BODY = "Hello! This is a test email sent via Gmail SMTP."

def send_email(SUBJECT, BODY, TO_EMAIL):
    msg = MIMEText(BODY, "plain")
    msg["Subject"] = SUBJECT
    msg["From"] = FROM_EMAIL
    msg["To"] = TO_EMAIL

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(USERNAME, PASSWORD)
        server.sendmail(FROM_EMAIL, TO_EMAIL, msg.as_string())

    print("Email sent!")

# send_email()
