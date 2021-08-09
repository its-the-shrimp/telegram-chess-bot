import redis
import telegram as tg
import telegram.ext


class RedisInterface(redis.Redis):
    def is_anon(self, user: tg.User) -> bool:
        return bool(self.exists(f"user-{user.id}:isanon"))

    def get_name(self, user: tg.User) -> str:
        if self.is_anon(user):
            return user.full_name
        else:
            return user.name

    def anon_mode_off(self, user: tg.User) -> None:
        self.delete(f"user-{user.id}:isanon")

    def anon_mode_on(self, user: tg.User) -> None:
        self.set(f"user-{user.id}:isanon", b"1")


class RedisPersistence(tg.ext.DictPersistence):
    USER_DATA = "ptb:user-data"
    CHAT_DATA = "ptb:chat-data"
    BOT_DATA = "ptb:bot-data"
    CALLBACK_DATA = "ptb:callback-data"
    CONVERSATIONS = "ptb:conversations"

    def __init__(
        self,
        url: str = None,
        db: RedisInterface = None,
        preprocessor=lambda self: None,
        **kwargs,
    ) -> None:

        if url is not None:
            self.conn = RedisInterface.from_url(url)
        elif db is not None:
            self.conn = db
        else:
            raise ValueError("Either 'url' or 'db' argument must be specified")

        super().__init__(
            **kwargs,
            user_data_json=self.conn.get(self.USER_DATA).decode()
            if self.conn.exists(self.USER_DATA)
            else "",
            chat_data_json=self.conn.get(self.CHAT_DATA).decode()
            if self.conn.exists(self.CHAT_DATA)
            else "",
            bot_data_json=self.conn.get(self.BOT_DATA).decode()
            if self.conn.exists(self.BOT_DATA)
            else "",
            callback_data_json=self.conn.get(self.CALLBACK_DATA).decode()
            if self.conn.exists(self.CALLBACK_DATA)
            else "",
            conversations_json=self.conn.get(self.CONVERSATIONS).decode()
            if self.conn.exists(self.CONVERSATIONS)
            else "",
        )
        self.preprocessor = preprocessor

    def flush(self) -> None:
        self.preprocessor(self)
        if self.store_bot_data:
            self.conn.set(self.BOT_DATA, self.bot_data_json.encode())
        if self.store_chat_data:
            self.conn.set(self.CHAT_DATA, self.chat_data_json.encode())
        if self.store_user_data:
            self.conn.set(self.USER_DATA, self.user_data_json.encode())
        if self.store_callback_data:
            self.conn.set(self.CALLBACK_DATA, self.callback_data_json.encode())
        if self._conversations_json or self._conversations:
            self.conn.set(self.CONVERSATIONS, self.conversations_json.encode())


class RedisContext(tg.ext.CallbackContext):
    @property
    def db(self):
        return self.dispatcher.persistence.conn
