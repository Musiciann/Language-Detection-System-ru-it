import os
import sys
import webbrowser
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import HOST, PORT


def main():
    from web.server import make_server

    server = make_server(HOST, PORT)
    url = f"http://{HOST}:{PORT}/"

    print("=" * 60)
    print(" Определитель языка текста — вариант 5")
    print(" Русский / Итальянский · N-грамм / алфавитный / нейросетевой (MLP) / ансамбль")
    print("=" * 60)
    print(f"\n  Сервер запущен:  {url}")
    print("  Остановить:      Ctrl+C\n")

    def _open_browser():
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Timer(0.6, _open_browser).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстанавливаюсь…")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
