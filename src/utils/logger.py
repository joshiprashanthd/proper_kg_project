import logging
from logging.handlers import RotatingFileHandler 


logger = logging.getLogger()
handler = RotatingFileHandler('logs.log', maxBytes=10000000, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)