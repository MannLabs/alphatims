#!python

# builtin
import setuptools
# local
import alphatims as package2install


with open("README.md", "r") as readme_file:
    LONG_DESCRIPTION = readme_file.read()

extra_requirements = {}
for extra, requirement_file_name in package2install.__requirements__.items():
    with open(requirement_file_name) as requirements_file:
        extra_requirements[extra] = []
        for line in requirements_file:
            if package2install.__requirements_style__ is None:
                extra_requirements[extra].append(line)
            else:
                requirement, version = line.split("==")
                requirement == requirement.strip()
                version == version.strip()
                if package2install.__requirements_style__ != "":
                    requirement = (
                        f"{requirement}"
                        f"{package2install.__requirements_style__}{version}"
                    )
                extra_requirements[extra].append(requirement)

requirements = extra_requirements.pop("")

setuptools.setup(
    name=package2install.__project__,
    version=package2install.__version__,
    license=package2install.__license__,
    description=package2install.__description__,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=package2install.__author__,
    author_email=package2install.__author_email__,
    url=package2install.__github__,
    project_urls=package2install.__urls__,
    keywords=package2install.__keywords__,
    classifiers=package2install.__classifiers__,
    packages=[package2install.__project__],
    include_package_data=True,
    entry_points={
        "console_scripts": package2install.__console_scripts__,
    },
    install_requires=requirements + [
        # TODO Remove hardcoded requirement?
        "pywin32==225; sys_platform=='win32'"
    ],
    extras_require=extra_requirements,
    python_requires=package2install.__python_version__,
)
