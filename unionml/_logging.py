import logging

logger = logging.getLogger("unionml")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[unionml] %(asctime)s %(name)s %(levelname)s %(message)s"))
logger.addHandler(handler)
