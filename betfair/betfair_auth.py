"""
Betfair Exchange API — authentication and session management.

Setup (one-time):
  1. pip install betfairlightweight
  2. Generate SSL cert:
       openssl genrsa -out certs/client-2048.key 2048
       openssl req -new -x509 -days 3650 -key certs/client-2048.key -out certs/client-2048.crt
  3. Upload certs/client-2048.crt at:
       betfair.com → My Account → Security → API Authentication → Add Certificate
  4. Set BETFAIR_PASSWORD in .env

Usage:
  from betfair_auth import get_client
  client = get_client(live=False)   # delayed key (read-only, no bets)
  client = get_client(live=True)    # live key (place bets)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ROOT = Path(__file__).resolve().parent.parent
CERTS_DIR = ROOT / "certs"

BETFAIR_USERNAME   = os.getenv("BETFAIR_USERNAME", "")
BETFAIR_PASSWORD   = os.getenv("BETFAIR_PASSWORD", "")
APP_KEY_LIVE       = os.getenv("BETFAIR_APP_KEY_LIVE", "")
APP_KEY_DELAY      = os.getenv("BETFAIR_APP_KEY_DELAY", "")
CERT_PATH          = os.getenv("BETFAIR_CERT_PATH", str(CERTS_DIR / "client-2048.crt"))
KEY_PATH           = os.getenv("BETFAIR_KEY_PATH",  str(CERTS_DIR / "client-2048.key"))


def get_client(live: bool = False):
    """Return an authenticated betfairlightweight client."""
    try:
        import betfairlightweight
    except ImportError:
        raise ImportError("Run: pip install betfairlightweight")

    if not BETFAIR_PASSWORD:
        raise ValueError("Set BETFAIR_PASSWORD in .env")
    if not Path(CERT_PATH).exists():
        raise FileNotFoundError(
            f"SSL cert not found at {CERT_PATH}\n"
            "Generate with:\n"
            "  mkdir certs\n"
            "  openssl genrsa -out certs/client-2048.key 2048\n"
            "  openssl req -new -x509 -days 3650 -key certs/client-2048.key -out certs/client-2048.crt\n"
            "Then upload certs/client-2048.crt to betfair.com → My Account → Security → API Authentication"
        )

    app_key = APP_KEY_LIVE if live else APP_KEY_DELAY
    # betfairlightweight expects the certs directory — it looks for
    # client-2048.crt and client-2048.key inside it automatically
    client = betfairlightweight.APIClient(
        username=BETFAIR_USERNAME,
        password=BETFAIR_PASSWORD,
        app_key=app_key,
        certs=str(CERTS_DIR),
    )
    client.login()
    return client


def test_connection(live: bool = False):
    """Quick connection test — prints account balance."""
    client = get_client(live=live)
    funds = client.account.get_account_funds()
    key_type = "LIVE" if live else "DELAYED"
    print(f"[OK] Connected ({key_type} key)")
    print(f"     Available balance: £{funds.available_to_bet_balance:.2f}")
    print(f"     Exposure:          £{funds.exposure:.2f}")
    return client


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Use live key instead of delayed")
    args = parser.parse_args()
    test_connection(live=args.live)
