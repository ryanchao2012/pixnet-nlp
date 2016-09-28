from celery import Celery
import re
import codecs
from celery.signals import worker_process_init, worker_init
from celery import Task
import logging

SEN_POOL = ''


@worker_process_init.connect
def load_sentences(sender=None, body=None, **kwargs):
    global SEN_POOL
    print('init')
    with codecs.open('../data/content-cn-lower.txt', 'r', encoding='utf-8') as f:
        outer_buf = []
        inner_buf = [''] * 10000
        buf_idx = 0
        for line in f:
            inner_buf[buf_idx] = line
            buf_idx += 1
            if buf_idx >= 10000: 
                buf_idx = 0
                outer_buf.append(''.join(inner_buf))
        outer_buf.append(''.join(inner_buf[:buf_idx]))
    SEN_POOL = ''.join(outer_buf)


BROKER_URL = 'amqp://guest:guest@localhost:5672//'
app = Celery(
    'tasks', broker=BROKER_URL, backend='amqp://', CELERY_RESULT_BACKEND='amqp')




@app.task
def parallel_retrieve(q):
    logging.warning('apply')
    match = re.search(q, SEN_POOL)
    if match: return match.group(1)
    else: return ''




