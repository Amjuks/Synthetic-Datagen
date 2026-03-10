import importlib.util


def has_package(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None
