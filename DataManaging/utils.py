import functools
import logging
import enum
import queue
import sys
import typing

from threading import Thread, Event


class SQLInsertFlags(enum.Enum):
    NO_PARENT = 1
    PARENT = 2
    REPLACE_COMMENT = 3


class ThreadStates(enum.Enum):
    STOPPED = -1
    FINISHED = 0
    RUNNING = 1


# Adapted from: https://stackoverflow.com/a/11420680/12921510
def catch_exception(f, logger: logging.Logger, handler=None):
    """
    :param f:
        function to wrap
    :param logger: logging.Logger
        Logger in which to send error message. Can not be none
    :param handler:
        Handler function for error. Can be None.
    :return:
    """

    @functools.wraps(f)
    def func(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f'Caught an exception in {f.__name__} Exception: {str(e)}')
            if handler is not None:
                handler(e)

    return func


class ArgumentError(Exception):
    pass


class StoppableThread(Thread):
    def __init__(self, target, bucket: queue.Queue, args: typing.Iterable = None, *other_args, **kwargs):
        super(StoppableThread, self).__init__(target=target, args=args, *other_args, **kwargs)
        self._stop_event = Event()
        self.bucket = bucket

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        try:
            super(StoppableThread, self).run()
        except Exception as e:
            self.bucket.put(sys.exc_info())


# Adapted from: https://stackoverflow.com/a/11420895/12921510
class ErrorCatcher(type):
    logger = logging.getLogger(__file__)
    handler = None

    def __new__(mcs, name, bases, dct):
        for m in dct:
            if hasattr(dct[m], '__call__'):
                dct[m] = catch_exception(dct[m], logger=mcs.logger, handler=mcs.handler)
        return type.__new__(mcs, name, bases, dct)
