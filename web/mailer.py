# web/mailer.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

LAB_EMAIL = ""
LAB_PASSWORD = ""  # Gmail App Password


def send_done_mail(user_email, job_id):
    if not user_email:
        print("[EMAIL] No user email provided, skip sending mail.")
        return False

    subject = f"[scGHSOM] Job {job_id} Completed"

    result_url = f"https://scghsom.changlabtw.com/results?job_id={job_id}"

    body = f"""
Hello,<br><br>

Your scGHSOM analysis has been completed.<br><br>

Job ID: {job_id}<br><br>

You can view the results at the following link:<br>
<a href="{result_url}">{result_url}</a><br><br>

Please note that results are deleted at 00:00 UTC every Monday.<br><br>

Thank you for using the scGHSOM web server.<br>
If you have any questions or encounter any issues, please feel free to contact us.<br><br>

Best regards,<br>
scGHSOM Web Server Team
"""

    msg = MIMEMultipart()
    msg["From"] = LAB_EMAIL
    msg["To"] = user_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "html"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(LAB_EMAIL, LAB_PASSWORD)
        server.send_message(msg)
        server.quit()

        print(f"[EMAIL SENT] → {user_email}")
        return True

    except Exception as e:
        print(f"[EMAIL ERROR] {e}")
        return False


