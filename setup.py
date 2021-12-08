import setuptools

with open("README", "r") as fh:
    long_description = fh.read()

with open('LICENSE') as fh:
    license = fh.read()
    
setuptools.setup(
    name="mytbl",
    version="1.0.0",
    author="CHEN Yongxin",
    author_email="chen_yongxin@outlook.com; Dr.Chen.Yongxin@qq.com",
    description="Python and shell scripts to handle data specifically from my turbulent boundary layer simulations",
    long_description=long_description,
    license=license,
    url="https://github.com/chenyongxin/mytbl",
    packages=setuptools.find_packages(exclude=("bin",)),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
