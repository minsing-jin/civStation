"""Console script for computer_use_test."""

import typer
from rich.console import Console

from computer_use_test import utils

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for computer_use_test."""
    console.print("Replace this message by putting your code into "
               "computer_use_test.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


if __name__ == "__main__":
    app()
