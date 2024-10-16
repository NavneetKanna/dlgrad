from enum import Enum, auto


class Device(Enum):
    CPU = auto()

    @staticmethod
    def from_str(d: str):
        try:
            return Device[d.upper()]
        except KeyError:
            print(f"Invalid device: {d}")
