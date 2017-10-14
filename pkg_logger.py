import logging
from os import path
from datetime import datetime

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%I:%M:%S %p',
                    filename=path.join(path.dirname(path.abspath(__file__)),'./test/logs/run-{}.log'.format(str(datetime.now()))),
                    filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)