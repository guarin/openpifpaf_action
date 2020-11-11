from setuptools import setup, find_packages

setup(
    name="openpifpaf_action_prediction",
    version="0.1.0",
    packages=find_packages(
        include=["openpifpaf_action_prediction", "openpifpaf_action_prediction.*"]
    ),
    install_requires=["openpifpaf"],
    extras_require={"dev": ["black"]},
)
