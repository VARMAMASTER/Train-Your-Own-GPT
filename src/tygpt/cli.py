"""Command-line entry point: `tygpt <command> ...`."""

import argparse
import sys
from typing import Optional

import yaml

from tygpt import __version__
from tygpt.config import ConfigError, config_to_dict, load_config


def cmd_config(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.overrides)
    print(yaml.safe_dump(config_to_dict(cfg), sort_keys=False), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tygpt", description="Train Your Own GPT")
    parser.add_argument("--version", action="version", version=f"tygpt {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    config = commands.add_parser(
        "config", help="load a config, apply overrides and print the resolved result"
    )
    config.add_argument("config", help="path to a YAML config file")
    config.add_argument("overrides", nargs="*", help="overrides like train.lr=3e-4")
    config.set_defaults(func=cmd_config)

    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
