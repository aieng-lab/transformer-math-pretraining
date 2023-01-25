import time


class TimerError(Exception):
    """Exception for wrong usage of Timer class"""


class Timer:

    def __init__(self):
        self.total_time = 0
        self._start_time = None
        self.elapsed_time = None

    def reset(self):
        self.total_time = 0
        self._start_time = None
        self.elapsed_time = None

    def start(self):
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it.")

        self.elapsed_time = None
        self._start_time = time.perf_counter()

    def stop(self):
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it.")

        self.elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

    def add_to_total(self):
        if self.elapsed_time is not None:
            self.total_time += self.elapsed_time
        else:
            raise TimerError(f"No time available for adding")

    def print_total(self):
        print(f"Seconds: {self.total_time:0.4f}")
        print(f"Minutes: {(self.total_time / 60):0.4f}")
        print(f"Hours:   {(self.total_time / (60 * 60)):0.4f}")

    def print_elapsed(self):
        print(f"Seconds: {self.elapsed_time:0.4f}")
        print(f"Minutes: {(self.elapsed_time / 60):0.4f}")
        print(f"Hours:   {(self.elapsed_time / (60 * 60)):0.4f}")
