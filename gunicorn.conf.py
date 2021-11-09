import os
from pyngrok import ngrok
import json
import logging

debug_env_path = os.path.join(os.path.dirname(__file__), "debug_env.json")
workers = 1
if os.path.exists(debug_env_path):
    with open(debug_env_path) as r:
        os.environ.update(json.load(r))
    os.environ["HOST_URL"] = ngrok.connect(addr=os.environ["PORT"]).public_url.replace(
        "http://", "https://"
    )
bind = "0.0.0.0:" + os.environ["PORT"]

try:
    os.mkdir(os.path.join(os.path.dirname(__file__), "images", "temp"))
except FileExistsError:
    pass
    

def worker_exit(server, _):
    try:
        dispatcher = server.app.callable.dispatcher
    except AttributeError:
        logging.error("Forced shutdown: could not cache match data")
        exit(1)
    else:
        logging.info(f"Flushing matches: {dispatcher.bot_data['matches']}")
        dispatcher.bot_data["conn"]._flush_matches(dispatcher.bot_data["matches"])
        dispatcher.bot_data["group_thread"].stop()
        dispatcher.bot_data["pm_thread"].stop()

    for filename in os.listdir(os.path.join("images", "temp")):
        os.remove(os.path.join("images", "temp", filename))
