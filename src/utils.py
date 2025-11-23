def is_east_of(lon_a, lon_b, allow_wrap=True):
    if lon_b > lon_a:
        return True
    if allow_wrap:
        return (lon_a > 150) and (lon_b < -150)
    return False

def travel_time(rank, country_a, country_b, pop_b):
    base = {1: 2, 2: 4, 3: 8}[rank]
    extra = 0
    if country_a != country_b:
        extra += 2
    if pop_b and pop_b > 200_000:
        extra += 2
    return base + extra
