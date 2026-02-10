"""Setup configuration for newsletter_agent_core package."""

from setuptools import setup, find_packages

setup(
    name="newsletter_agent_core",
    version="0.1.0",
    description="Newsletter Agent with clustering and axiom-based filtering",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "google-api-python-client>=2.0.0",
        "google-auth-oauthlib>=0.5.0",
        "google-auth-httplib2>=0.1.0",
        "google-auth>=2.0.0",
        "google-generativeai>=0.3.0",
        "beautifulsoup4>=4.9.0",
        "requests>=2.25.0",
        "langchain-text-splitters>=0.0.1",
        "python-dotenv>=0.19.0",
        "pandas>=1.5.0",
    ],
    extras_require={
        "clustering": [
            "sentence-transformers>=2.2.2",
            "hdbscan>=0.8.29",
            "scikit-learn>=1.3.0",
            "numpy>=1.21.0",
            "scipy>=1.9.0",
            "psutil>=5.9.0",
        ],
    },
)
