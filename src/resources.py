from contextlib import contextmanager
from threading import Condition
import time


class Resource:
    def __init__(self, name: str):
        self.name = name
        self.condition = Condition()
        self.issued = 0
        self.serving = 0

    @contextmanager
    def use(self, timing):
        started = time.perf_counter()
        with self.condition:
            ticket = self.issued
            self.issued += 1
            self.condition.wait_for(lambda: ticket == self.serving)
        try:
            timing.wait(self.name, time.perf_counter() - started)
            yield
        finally:
            with self.condition:
                self.serving += 1
                self.condition.notify_all()
