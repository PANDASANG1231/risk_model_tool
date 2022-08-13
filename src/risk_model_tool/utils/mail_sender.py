# coding: utf-8
# Author: Wenjia Zhu

import sys
import time
import html
import logging
import smtplib
import traceback

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage


class MailSender(object):
    
    def __init__(self, config, html):
        self.host = config['host']
        self.port = config['port']
        self.user = config['user']
        self.passwd = config.get('passwd')
        self.sub = config['subject']
        self.to = config['to']
        self.timeout = config['timeout']
        self.msgRoot = MIMEMultipart('related')
        self.html = html
    
    def send(self):
        html_content=self.html
        msg_text = MIMEText(html_content, 'html', 'utf-8')
        self.msgRoot.attach(msg_text)
        self.msgRoot['Subject'] = self.sub
        self.msgRoot['From'] = self.user
        self.msgRoot['To'] = ';'.join(self.to)
        self.msgRoot.preamble = 'This is a multi-part message in MIME format.'

        for i in range(4):
            try :
                s = smtplib.SMTP(self.host, self.port, timeout=self.timeout)
                s.ehlo()
                s.starttls() #Puts connection to SMTP server in TLS mode
                s.ehlo()
                s.login(self.user, self.passwd)
                s.sendmail(self.user, self.to,  msg=self.msgRoot.as_string())
                s.close()
                logging.info('Mail sending done')
                break
            except Exception as e:
                if i < 3:
                    logging.info('Cannot connect to the SMTP, sleep 60s and retry the %s time' % str(i+1))
                    time.sleep(1)
                else:
                    logging.info('Already retried 3 times')
                    logging.info('Mail sending failed')
                    logging.error(traceback.format_exc())
                pass


