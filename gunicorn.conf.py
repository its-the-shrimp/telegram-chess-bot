import os
import multiprocessing

bind = ":".join([os.environ["HOST_URL"], os.emviron["PORT"]])
workers = multiprocessing.cpu_count() * 2 + 1