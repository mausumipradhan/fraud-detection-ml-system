from setuptools import setup, find_packages

setup(
    name="fraud-detection",
    version="1.0.0",
    description="Real-Time Credit Card Fraud Detection System",
    author="Mausumi Pradhan",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=open("requirements.txt").read().splitlines(),
)
