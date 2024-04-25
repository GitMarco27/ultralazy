from setuptools import setup


def read_requirements():
    with open("requirements.txt", "r") as req:
        content = req.read()
        requirements = content.split("\n")

    return requirements


setup(
    name="ultralazy",
    version="0.0.1",
    packages=[
        "ultralazy",
        "ultralazy.utils"
    ],
    url="https://github.com/GitMarco27/ultralazy.git",
    license="MIT",
    author="Marco Sanguineti",
    author_email="marco.sanguineti.info@gmail.com",
    description="Ultralazy",
    install_requires=read_requirements(),
    include_package_data=True,  # This tells setuptools to include files specified in MANIFEST.in
)
