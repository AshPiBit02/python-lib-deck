from enum import Enum

class HLOrder(str,Enum):
    high_to_low="HighToLow"
    low_to_high="LowToHigh"

class ExtremeValue(str,Enum):
    highest="Highest"
    lowest="Lowest"

