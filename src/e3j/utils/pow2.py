def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()
