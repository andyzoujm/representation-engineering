from setuptools import setup, find_namespace_packages

setup(
    name="repe",
    version="0.1",
    description="",
    packages=["repe"],
    author="Center for AI Safety",
    author_email="",
    url="https://github.com/yourusername/your_project_name",  # if you have a github repo
    install_requires=[
        "git+https://github.com/huggingface/transformers.git",
        "git+https://github.com/huggingface/accelerate.git",
    ],
)
