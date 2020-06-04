# mtspy

![CI](https://github.com/IgorBaratta/mtspy/workflows/CI/badge.svg)
![CI Docker images](https://img.shields.io/docker/cloud/build/igorbaratta/mtspy)
![stability-wip](https://img.shields.io/badge/stability-work_in_progress-lightgrey.svg)

Multi-threaded sparse matrix operations in Python

## Installation

Install with:

```shell
pip3 install git+https://github.com/IgorBaratta/mtspy.git --upgrade
```

or if the anot ll requirements are installed:

```shell
git clone --recursive https://github.com/IgorBaratta/mtspy.git
cd mtspy
python3 -m pip install -r  requirements.txt
python3 -m pip install .
```

### Requirements

Make sure to clone with **--recursive** to download the required submodules!

- Numpy
- Scipy
- C++ 17 compiler with OpenMP support
- Eigen3
- pybind11

Using docker container with all requirements installed:

```shell
docker pull igorbaratta/mtspy:latest
```

## License

 **mtspy** is licensed under the GNU General Public License v3.0 -- see the [LICENSE](LICENSE) file for details
