import pytest
from _pytest.config import Config, Parser
import tensorflow as tf


def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--eager",
        action="store_true",
        default=False,
        help="whether to run all functions eagerly",
    )

def pytest_collection_modifyitems(config: Config, items: list[pytest.Item]) -> None:
    if config.getoption("--eager"):
        tf.config.experimental_run_functions_eagerly(True)  
def pytest_collection_modifyitems(config: Config, items: list[pytest.Item]) -> None:
    if config.getoption("--eager"):
        import tensorflow as tf
        tf.config.experimental_run_functions_eagerly(True)