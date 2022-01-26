import smtplib
import ssl


class Mailer:

    def __init__(self):

        self.EMAIL = ""

        self.PASS = ""
        self.PORT = 465
        self.server = smtplib.SMTP_SSL('smtp.gmail.com', self.PORT)

    def send(self, mail):
        self.server = smtplib.SMTP_SSL('smtp.gmail.com', self.PORT)
        self.server.login(self.EMAIL, self.PASS)
        # message to be sent
        SUBJECT = 'ALERT!'
        TEXT = f'People limit exceeded in your building!'
        message = 'Subject: {}\n\n{}'.format(SUBJECT, TEXT)

        # sending the mail
        self.server.sendmail(self.EMAIL, mail, message)
        self.server.quit()
