"""Entry point for `python -m blochspheresim`."""

from .app import App


def main() -> None:
    """Create and run the simulator."""
    App().run()


if __name__ == "__main__":
    main()
