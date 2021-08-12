import redis
from telegram.ext import DictPersistence, CallbackContext
import logging


class RedisInterface(redis.Redis):
    def is_anon(self, user) -> bool:
        return bool(self.exists(f"user-{user.id}:isanon"))

    def get_name(self, user) -> str:
        if self.is_anon(user):
            return f"player{user.id}"
        else:
            return user.name

    def anon_mode_off(self, user) -> None:
        self.delete(f"user-{user.id}:isanon")

    def anon_mode_on(self, user) -> None:
        self.set(f"user-{user.id}:isanon", b"1")


class RedisPersistence(DictPersistence):
    USER_DATA = "ptb:{token}:user-data"
    CHAT_DATA = "ptb:{token}:chat-data"
    BOT_DATA = "ptb:{token}:bot-data"
    CALLBACK_DATA = "ptb:{token}:callback-data"
    CONVERSATIONS = "ptb:{token}:conversations"

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
    @property
    def db(self):
        return self.dispatcher.persistence.conn
