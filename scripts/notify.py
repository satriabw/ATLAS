import os
import urllib.parse
import requests


def send_whatsapp(message: str) -> None:
    phone  = os.environ.get("CALLMEBOT_PHONE")
    apikey = os.environ.get("CALLMEBOT_APIKEY")
    if not phone or not apikey:
        return
    url = (
        "https://api.callmebot.com/whatsapp.php"
        f"?phone={phone}&text={urllib.parse.quote(message)}&apikey={apikey}"
    )
    try:
        requests.get(url, timeout=10)
    except Exception:
        pass


if __name__ == "__main__":
    import sys
    send_whatsapp(sys.argv[1] if len(sys.argv) > 1 else "test")
