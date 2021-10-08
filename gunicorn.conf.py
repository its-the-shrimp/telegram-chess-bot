import os
import multiprocessing

bind = "0.0.0.0:"+os.environ["PORT"]
workers = 1