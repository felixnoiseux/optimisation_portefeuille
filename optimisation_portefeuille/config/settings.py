from dataclasses import dataclass


@dataclass
class Settings:
    # Data
    years_history: int = 5
    trading_days: int = 252

    # Constraints
    weight_min: float = 0.00
    weight_max: float = 0.20

    # Backtesting / turnover
    turnover_max: float = 0.20

    # Risk-free rate (annualized)
    risk_free_rate: float = 0.02

    # Logging
    log_level: str = "INFO"


def get_settings() -> Settings:
    return Settings()
