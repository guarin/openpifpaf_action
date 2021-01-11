from setuptools import setup, find_packages

setup(
    name="openpifpaf_action",
    version="0.1.0",
    packages=find_packages(include=["openpifpaf_action", "openpifpaf_action.*"]),
    install_requires=["openpifpaf"],
    extras_require={"dev": ["black"]},
)
