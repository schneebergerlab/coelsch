import os
import sys
import logging
from datetime import datetime
from contextlib import contextmanager
from collections import deque

import click


LOG_FORMATTING = {
    'INFO': dict(fg='green', label='INFO'),
    'ERROR': dict(fg='red', label='ERROR'),
    'EXCEPTION': dict(fg='red', label='EXCPT'),
    'CRITICAL': dict(fg='red', label='CRIT'),
    'DEBUG': dict(fg='blue', label='DEBUG'),
    'WARNING': dict(fg='yellow', label='WARN')
}


def log_msg_formatter(tool_name=None):
    if tool_name is None:
        tool_name = os.path.split(sys.argv[0])[1]
    tool_name = tool_name.upper()
    def _format_log_msg(lvl, msg):
        dt = datetime.now().strftime("%Y%m%d:%H:%M:%S")
        fmt = LOG_FORMATTING[lvl]
        label = fmt['label'].rjust(5)
        prefix = click.style(f'[{dt}] {tool_name} {label}:', fg=fmt['fg'])
        return f'{prefix} {msg}'
    return _format_log_msg


class LogFilter(logging.Filter):

    def __init__(self, memory=10):
        super().__init__()
        self.msgs = deque(maxlen=memory)

    def filter(self, record):
        '''only display new messages, prevents extreme repetition of warnings'''
        msg = record.getMessage()
        rv = msg not in self.msgs
        if rv:
            self.msgs.append(msg)
        return rv


class LogFormatter(logging.Formatter):

    def __init__(self, tool_name):
        super().__init__()
        self.format_log_msg = log_msg_formatter(tool_name)

    def format(self, record):
        if not record.exc_info:
            lvl = record.levelname
            msg = record.getMessage()
            msg = '\n'.join(self.format_log_msg(lvl, line) for line in msg.splitlines())
            return msg
        return logging.Formatter.format(self, record)


class ClickLogHandler(logging.Handler):

    def emit(self, record):
        try:
            msg = self.format(record)
            click.echo(msg, err=True)
            if record.levelno >= logging.ERROR:
                click.echo(format_log_msg(record.levelname, 'Unable to continue, exiting...'), err=True)
                sys.exit(1)
        except Exception:
            self.handleError(record)


def click_logger(tool_name):
    def _click_logger(ctx, params, value):
        log = logging.getLogger(tool_name)
        log.setLevel(value)
        log_filter = LogFilter(memory=10)
        log.addFilter(log_filter)
        log_handler = ClickLogHandler()
        log_handler.formatter = LogFormatter(tool_name)
        log.handlers = [log_handler]
        log.propagate = False
        return log
    return _click_logger


def progress_bar(iterable, label=None, **kwargs):
    if label is not None:
        label = log_msg_formatter()('INFO', label)
    default_kwargs = {'width': 25, 'show_eta': False, 'show_percent': False}
    default_kwargs.update(kwargs)
    return click.progressbar(iterable, label=label, **default_kwargs)
