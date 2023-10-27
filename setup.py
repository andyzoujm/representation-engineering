from setuptools import setup, find_namespace_packages

setup(
    name="repe",
    version="0.1",
    description="",
    packages=find_namespace_packages(),  # This will automatically find packages
    author="Center for AI Safety",
    author_email="",
    url="https://github.com/andyzoujm/representation-engineering",
    install_requires=[
        "transformers",
        "accelerate",
    ],
)
