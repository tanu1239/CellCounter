from setuptools import setup, find_packages
import pathlib
from importlib import import_module

# Import version from the package
version = "0.1.0"

# Dynamically load other project metadata from pyproject.toml
here = pathlib.Path(__file__).parent
pyproject_path = here / "pyproject.toml"
try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib

data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
meta = data.get("project", {})

setup(
    name=meta.get("name"),
    version=version,
    description=meta.get("description", ""),
    long_description=(here / meta.get("readme", "README.md")).read_text(
        encoding="utf-8"
    ),
    long_description_content_type="text/markdown",
    author=", ".join(a.get("name", "") for a in meta.get("authors", [])),
    author_email=", ".join(a.get("email", "") for a in meta.get("authors", [])),
    url=meta.get("urls", {}).get("homepage", meta.get("url", "")),
    license=meta.get("license", {}).get("text", ""),
    classifiers=meta.get("classifiers", []),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=meta.get("requires-python"),
    install_requires=meta.get("dependencies", []),
    extras_require=meta.get("optional-dependencies", {}),
    include_package_data=True,
)
