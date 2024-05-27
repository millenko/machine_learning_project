from time import time

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        time_taken = int(end_time - start_time)
        return result, time_taken
    return wrapper