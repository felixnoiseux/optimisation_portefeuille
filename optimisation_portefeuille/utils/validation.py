from typing import List


def validate_tickers(tickers: List[str]) -> List[str]:
    cleaned = []
    for t in tickers:
        t = t.strip().upper()
        if t:
            cleaned.append(t)
    if not cleaned:
        raise ValueError("Aucun symbole fourni. Veuillez spécifier au moins un ticker.")
    return list(dict.fromkeys(cleaned))  # unique & preserve order
