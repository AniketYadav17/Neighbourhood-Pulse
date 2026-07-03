"""Command-line entry point: `pulse <stage>`."""

import argparse
import logging
import sys


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        stream=sys.stdout,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pulse", description="Neighbourhood Pulse data pipeline")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="fetch planning applications and cafés (resumable)")
    sub.add_parser("transform", help="filter coordinates, convert CRS, assign H3 hexagons")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    # Imports deferred so `pulse --help` stays fast and tests can stub the classes.
    if args.command == "ingest":
        from neighbourhood_pulse.ingestion import DataIngestion, IngestionError

        try:
            DataIngestion().run()
        except IngestionError as e:
            logging.getLogger(__name__).error("Ingestion failed: %s", e)
            raise SystemExit(1) from e
    elif args.command == "transform":
        from neighbourhood_pulse.transformation import DataTransformation

        DataTransformation().run()
