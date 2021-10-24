import os
import multiprocessing

bind = "0.0.0.0:"+os.environ["PORT"]
workers = multiprocessing.cpu_count() * 2 + 1
