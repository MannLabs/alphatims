#!python

# builtin
import setuptools
# local
import alphatims as package_to_install


with open("README.md", "r") as readme_file:
    LONG_DESCRIPTION = readme_file.read()


with open("requirements.txt") as requirements_file:
    requirements = []
    for line in requirements_file:
        if package_to_install.__strict_requirements__:
            requirement = line.strip()
        else:
            requirement, version = line.split("==")
        requirements.append(requirement)


extra_requirements = {}
for extra, file_name in package_to_install.__extra_requirements__.items():
    with open(file_name) as requirements_file:
        extra_requirements[extra] = []
        for line in requirements_file:
            if package_to_install.__strict_requirements__:
                requirement = line.strip()
            else:
                requirement, version = line.split("==")
            extra_requirements[extra].append(requirement)


setuptools.setup(
    name=package_to_install.__project__,
    version=package_to_install.__version__,
    license=package_to_install.__license__,
    description=package_to_install.__description__,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=package_to_install.__author__,
    author_email=package_to_install.__author_email__,
    url=package_to_install.__github__,
    project_urls=package_to_install.__urls__,
    keywords=package_to_install.__keywords__,
    classifiers=package_to_install.__classifiers__,
    packages=setuptools.find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": package_to_install.__console_scripts__,
    },
    install_requires=requirements + [
        # TODO Remove hardcoded requirement?
        "pywin32==225; sys_platform=='win32'"
    ],
    extras_require=extra_requirements,
    python_requires=package_to_install.__python_version__,
)
