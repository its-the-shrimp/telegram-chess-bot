import os
from typing import Callable, Generator, Iterator, Optional
import redis
import sys
import json
import gzip
import chess as core
from telegram import InlineKeyboardMarkup, InlineKeyboardButton, Update, User, Bot
from telegram.ext import Dispatcher, CallbackContext


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in Singleton._instances:
            Singleton._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return Singleton._instances[cls]

class RedisInterface(redis.Redis, metaclass=Singleton):

    def _fetch_matches(self, decoder: Callable[[bytes], object], bot: Bot) -> Generator[tuple[int, bytes], None, None]:
        for key in self.scan_iter(match="match:*"):
            mid = key.split(b":")[1].decode()
            yield mid, decoder(self.get(key), bot, mid)

    def _flush_matches(self, matches: dict[str, object]) -> None:
        for id, matchobj in matches.items():
            self.set(f"match:{id}", bytes(matchobj))

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

    def get_user_ids(self) -> Generator[int, None, None]:
        for key in self.scan_iter(match="*:lang"):
            key = key.decode()
            yield int(key[: key.find(":")])

    def get_langcodes_stats(self) -> dict[str, int]:
        counter = {}
        for key in self.keys(pattern="*:lang"):
            key = self.get(key).decode()
            if key in counter:
                counter[key] += 1
            else:
                counter[key] = 1

        return counter

class OptionValue:
    def __init__(self, value: str, condition: Callable[[dict[str, str]], bool] = None):
        self.value = value 
        self.is_available = condition or (lambda _: True)

    def __repr__(self):
        return f"<OptionValue({self.value})>"

    def __eq__(self, other):
        return isinstance(self, other.__class__) and self.value == other.value

class MenuOption:
    @classmethod
    def from_dict(cls, name: str, obj: dict) -> "MenuOption":
        return cls(
            name,
            [OptionValue(k, condition=v) for k, v in obj["values"].items()],
            condition=obj.get("condition")
        )

    def __init__(self, name: str, values: list[OptionValue], condition: Callable[[dict[str, str]], bool] = None):
        self.name = name
        self.values = values
        self.is_available = condition or (lambda _: True)

    def __repr__(self):
        return f"<MenuOption({self.name}); {len(self.values)} values>"

    def __iter__(self) -> Iterator[OptionValue]:
        return self.values.__iter__()

    def available_values(self, indexes: dict[str, str]) -> list[str]:
        if not self.is_available(indexes):
            return []
        return [i.value for i in self.values if i.is_available(indexes)]


class MenuFormatter(object, metaclass=Singleton):

    @classmethod
    def from_dict(cls, obj: dict) -> "MenuFormatter":
        return cls([MenuOption.from_dict(*args) for args in obj.items()])

    def __init__(self, options: list[MenuOption]):
        self.options: list[MenuOption] = options
        self.defaults: dict[str, int] = {}

        for option in self:
            if not self.defaults:
                self.defaults[option.name] = option.values[0].value
            else:
                self.defaults[option.name] = self.get_default_value(option.name, self.defaults)

    def __iter__(self) -> Iterator[MenuOption]:
        return self.options.__iter__()

    def __getitem__(self, key: str) -> MenuOption:
        for option in self:
            if option.name == key:
                return option

        raise KeyError(key)

    def get_value(self, key: str, index: int) -> str:
        lookup_list = [value.value for value in self[key]]
        return lookup_list[index]

    def get_index(self, key: str, value: str) -> int:
        lookup_list = [value.value for value in self[key]]
        return lookup_list.index(value)

    def get_default_value(self, name: str, indexes: dict[str, str]) -> str:
        try:
            return self[name].available_values(indexes)[0]
        except IndexError:
            return None

    def decode(self, keyboard: InlineKeyboardMarkup) -> dict[str, str]:
        res = {}
        for column in keyboard.inline_keyboard[:-1]:
            column_data = core.parse_callback_data(column[1].callback_data)["args"]
            res[column_data[0]] = self.get_value(column_data[0], column_data[1])

        return res

    def encode(self, user: User, indexes: dict[str, str] = None) -> InlineKeyboardMarkup:
        res = []
        indexes = indexes or self.defaults

        for option in self:
            if option.is_available(indexes):
                if option.name in indexes:
                    chosen = indexes[option.name]
                else:
                    chosen = self.get_default_value(option.name, indexes)
                    indexes[option.name] = chosen
                res.append(
                    [
                        InlineKeyboardButton(
                            text="◀️", callback_data=core.format_callback_data(
                                "PREV",
                                args=[option.name],
                                expected_uid=user.id,
                                handler_id="MAIN",

                            )
                        ),
                        InlineKeyboardButton(
                            text=core.langtable[user.language_code][chosen],
                            callback_data=core.format_callback_data(
                                "DESC",
                                args=[option.name, self.get_index(option.name, chosen)],
                                handler_id="MAIN",
                                expected_uid=user.id
                            ),
                        ),
                        InlineKeyboardButton(
                            text="▶️", callback_data=core.format_callback_data(
                                "NEXT",
                                args=[option.name],
                                handler_id="MAIN",
                                expected_uid=user.id
                            )
                        ),
                    ]
                )
            else:
                if option.name in indexes:
                    del indexes[option.name]

        inline_query = self.tg_encode(indexes) if indexes["mode"] == "invite" else None

        res.append(
            [
                InlineKeyboardButton(
                    text=core.langtable[user.language_code]["cancel-button"],
                    callback_data=core.format_callback_data(
                        "CANCEL",
                        handler_id="MAIN",
                        expected_uid=user.id
                    ),
                ),
                InlineKeyboardButton(
                    text=core.langtable[user.language_code]["play-button"],
                    switch_inline_query=inline_query,
                    callback_data=None if inline_query else core.format_callback_data(
                        "PLAY",
                        handler_id="MAIN",
                        expected_uid=user.id
                    ),
                ),
            ]
        )

        return InlineKeyboardMarkup(res)

    def prettify(self, indexes: dict[str, str], lang_code: str) -> str:
        locale = core.langtable[lang_code]

        return "\n".join([
            locale["options-title"],
            ", ".join([locale[i] for i in indexes.values()])
        ])

    def tg_encode(self, indexes: dict[str, str]) -> str:
        return " ".join([":".join((k, v)) for k, v in indexes.items() if k != "mode"])

    def tg_decode(self, src: str) -> dict[str, str]:
        indexes = dict(token.split(":") for token in src.split(" ") if ":" in token)
        for option in self:
            is_available = option.is_available(indexes)
            if option.name not in indexes and is_available:
                indexes[option.name] = self.get_default_value(option.name, indexes)
            elif option.name in indexes and not is_available:
                del indexes[option.name]

        return indexes        


class BoardGameContext(CallbackContext):
    menu: MenuFormatter = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.langtable: dict[str, str] = None
        RedisInterface.bot = self.dispatcher.bot

    @classmethod
    def from_update(cls, update: Update, dispatcher: Dispatcher) -> "BoardGameContext":
        self = super().from_update(update, dispatcher)
        if update.effective_user is not None:
            self.langtable = core.langtable[update.effective_user.language_code]
        
        self.db.set(f"{update.effective_user.id}:lang", update.effective_user.language_code.encode())

        return self

    @property
    def db(self) -> RedisInterface:
        return RedisInterface()


if __name__ == "__main__":
    print("Connecting to database...")
    env = json.load(open("debug_env.json"))
    conn = RedisInterface.from_url(env["REDISCLOUD_URL"])

    os.system("clear")
    print("Board Game Bot CLI\nEnter a command. To get list of commands, type 'help'")

    while True:
        try:
            command, *args = input(">>> ").split(" ")
        except:
            print("\nClosing connection to the database...")
            conn.close()
            break

        if command == "matches":
            count = 0
            memory_sum = 0
            for key in conn.scan_iter(match="match:*"):
                count += 1
                match = conn.get(key)
                memory_sum += len(match)
                print("Match", key.split(b":")[1].decode(), f"({len(match)} bytes)", ":")
                print(match.decode(errors="replace"), "\n")
            print(f"Total {count} matches ( {round(memory_sum / 1024, 2)} Kbytes) .")

        if command == "del-match":
            try:
                conn.delete(f"match:{args[0]}")
                print(f"Match {args[0]} deleted.")
            except KeyError:
                print("Match ID not specified.")

        if command == "user-stats":
            print("Number of users per language code:")
            allusers = 0
            for langcode, count in conn.get_langcodes_stats().items():
                allusers += (count)
                print(f"{langcode}: {count}")
            print(f"\nTotal {allusers} users.")

        if command == "help":
            print("""
List of commands:
    'matches' - Get information about all ongoing matches.
    'del-match [id]' - Delete a match with a specified ID.
    'user-stats' - Get statistics about users of the bot.
    'help' - Display this message.
""")
        else:
            print(f"unknown command: {command!r}\nTo get a list of all available commands, type 'help'")



