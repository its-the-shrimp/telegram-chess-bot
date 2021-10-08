from typing import Generator, Optional
import redis
from telegram import Bot, User
from telegram.ext import DictPersistence, CallbackContext
import logging
import sys
import json
import gzip


class RedisInterface(redis.Redis):
    def get_name(self, user: User) -> str:
        if self.get_anon_mode(user):
            return f"player{user.id}"
        else:
            return user.name

    def create_invite(self, invite_id: str, user: User, options: dict):
        self.set(
            f"invite:{invite_id}",
            gzip.compress(
                json.dumps({"from_user": user.to_dict(), "options": options}).encode()
            ),
        )

    def get_invite(self, invite_id: str) -> Optional[dict]:
        raw = self.get(f"invite:{invite_id}")
        if raw:
            self.delete(f"invite:{invite_id}")
            res = json.loads(gzip.decompress(raw).decode())
            res["from_user"] = User.de_json(res["from_user"], self.bot)
            return res

    def get_anon_mode(self, user: User) -> bool:
        return bool(self.exists(f"{user.id}:isanon"))

    def set_anon_mode(self, user: User, value: bool) -> None:
        if value:
            self.set(f"{user.id}:isanon", b"1")
        else:
            self.delete(f"{user.id}:isanon")

    def get_langcode(self, user: User) -> str:
        return (self.get(f"{user.id}:lang") or b"en").decode()

    def cache_user_data(self, user: User) -> None:
        self.set(f"{user.id}:lang", user.language_code.encode())

    def get_user_ids(self) -> Generator:
        for key in self.scan_iter(match="*:lang"):
            key = key.decode()
            yield int(key[: key.find(":")])


class RedisPersistence(DictPersistence):
    USER_DATA = "ptb:{token}:user-data"
    CHAT_DATA = "ptb:{token}:chat-data"
    BOT_DATA = "ptb:{token}:bot-data"
    CALLBACK_DATA = "ptb:{token}:callback-data"
    CONVERSATIONS = "ptb:{token}:conversations"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bot: Bot = None

    @staticmethod
    def default_decoder(self, obj):
        return {k: v.decode() for k, v in obj.items()}

    @staticmethod
    def default_encoder(self, obj):
        return {
            "bot_data": self.bot_data_json.encode(),
            "chat_data": self.chat_data_json.encode(),
            "user_data": self.user_data_json.encode(),
            "callback_data": self.callback_data_json.encode(),
            "conversations": self.conversations_json.encode()
            if self.conversations
            else b"",
        }

    def __init__(
        self,
        url: str = None,
        db: RedisInterface = None,
        decoder=None,
        encoder=None,
        **kwargs,
    ) -> None:

        if url is not None:
            self.conn = RedisInterface.from_url(url)
        elif db is not None:
            self.conn = db
        else:
            raise ValueError("Either 'url' or 'db' argument must be specified")

        self.decoder = decoder if decoder else self.default_decoder
        self.encoder = encoder if encoder else self.default_encoder
        self._init_kwargs = kwargs

    def set_bot(self, bot):
        obj = {}
        logging.debug("Fetching persistence data from the database...")
        for key in [
            "USER_DATA",
            "CHAT_DATA",
            "BOT_DATA",
            "CALLBACK_DATA",
            "CONVERSATIONS",
        ]:
            setattr(self, key, getattr(self, key).format(token=bot.token))
            if self._init_kwargs.get(f"store_{key.lower()}", True):
                obj[key.lower()] = (
                    self.conn.get(getattr(self, key))
                    if self.conn.exists(getattr(self, key))
                    else b""
                )
                logging.debug(
                    f"{getattr(self, key)}({round(len(obj[key.lower()])/1024, 2)}Kb): {obj[key.lower()]}"
                )

        obj = self.decoder(self, obj)
        super().__init__(
            **self._init_kwargs,
            user_data_json=obj.get("user_data"),
            chat_data_json=obj.get("chat_data"),
            bot_data_json=obj.get("bot_data"),
            callback_data_json=obj.get("callback_data"),
            conversations_json=obj.get("conversations"),
        )
        super().set_bot(bot)
        self.conn.bot = bot
        del self._init_kwargs

    def flush(self) -> None:
        logging.debug("Flushing data to the database...")
        obj = {
            k: getattr(self, k)
            for k in [
                "chat_data",
                "bot_data",
                "user_data",
                "callback_data",
                "conversations",
            ]
        }
        obj = self.encoder(self, obj)

        for k, v in obj.items():
            if getattr(self, f"store_{k}", True):
                self.conn[getattr(self, k.upper())] = v
                logging.debug(
                    f"{getattr(self, k.upper())}({round(len(obj[k.lower()])/1024, 2)}Kb): {v}"
                )


class RedisContext(CallbackContext):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.langtable: dict[str, str] = {}

    @property
    def db(self) -> RedisInterface:
        return self.dispatcher.persistence.conn


if __name__ == "__main__":
    env = json.load(open("debug_env.json"))
    conn = redis.Redis.from_url(env["REDISCLOUD_URL"])

    if sys.argv[1] == "clear":
        for key in conn.scan_iter():
            del conn[key]

    elif sys.argv[1] == "peek":
        res = {}
        for key in conn.scan_iter():
            key = key.decode()
            if conn.type(key) == b"string":
                try:
                    res[key] = gzip.decompress(conn.get(key)).decode()
                    res[key] = json.loads(res[key]) if res[key] else res[key]
                except:
                    res[key] = conn.get(key).decode()
            elif conn.type(key) == b"set":
                res[key] = [i.decode() for i in conn.sscan_iter(key)]

        print(json.dumps(res, indent=2))
