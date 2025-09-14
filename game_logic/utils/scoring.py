def points_to_value(label: str) -> int:
    return {"1PT": 1, "2PT": 2, "3PT": 3}.get(label, 0)

