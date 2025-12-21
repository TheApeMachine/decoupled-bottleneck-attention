class MathUtils:
    """
    MathUtils is a class that contains static methods for mathematical operations.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def bits(x: int) -> int:
        """
        bits returns the number of bits required to represent the integer x.
        """
        return max(1, int(x).bit_length())

    @staticmethod
    def clampf(x: float, lo: float, hi: float) -> float:
        """
        clampf clamps the float x between lo and hi.
        """
        return float(max(float(lo), min(float(x), float(hi))))