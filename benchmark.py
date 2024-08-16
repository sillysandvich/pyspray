#benchmark stuff
from time import time
def benchmark(func): 
    def timedfunc(*args, **kwargs):
        start_time = time()
        ret = func(*args, **kwargs)
        end_time = time()
        print(f"{func.__name__} took: {end_time - start_time} seconds")
        return ret
    return timedfunc