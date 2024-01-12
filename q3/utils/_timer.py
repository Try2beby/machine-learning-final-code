import timeit


class Timer:
    def __init__(self, name):
        self.name = name
        self.start = timeit.default_timer()

    def stop(self):
        self.end = timeit.default_timer()
        print(
            f"Time taken for {self.name}: {format(self.end - self.start, '.3g')} seconds"
        )
