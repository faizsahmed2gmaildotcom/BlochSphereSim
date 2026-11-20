"""Entry point for `python app.py`."""

from blochspheresim.app import App


def main() -> None:
    """Create and run the simulator."""
    App().run()


if __name__ == "__main__":
    main()
