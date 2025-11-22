import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
from datetime import datetime

def send_email():
    sender = os.getenv('GMAIL_USER')
    password = os.getenv('GMAIL_APP_PASSWORD')
    recipient = os.getenv('GMAIL_USER')
    
    if not sender or not password:
        print("❌ Gmail credentials not found")
        return False
    
    with open('email_report.html', 'r') as f:
        html_content = f.read()
    
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f'🤖 Forex AI Pipeline - 10 Run Progress Report ({datetime.now().strftime("%Y-%m-%d")})'
    msg['From'] = sender
    msg['To'] = recipient
    
    html_part = MIMEText(html_content, 'html')
    msg.attach(html_part)
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        
        print(f"✅ Email sent successfully to {recipient}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send email: {str(e)}")
        return False

if __name__ == "__main__":
    success = send_email()
    exit(0 if success else 1)
