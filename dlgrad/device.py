from enum import Enum, auto


class Device(Enum):
    CPU = auto()
    METAL = auto()

    @staticmethod
    def from_str(d: str) -> str:
        try:
            return Device[d.upper()]
        except KeyError:
            print(f"Invalid device: {d}")
