# HNFC Agent

This repository provides the backend code for a fake news detection framework. This application is described in the article "Building a framework for fake news detection in the health domain".

## Installation

This section describes the installation process of this framework. This has been tested in Linux environments such as Debian and Ubuntu.

### Prerequisites

It is necessary to install the [frontend environment](https://github.com/jrmtnez/hnfc-site) first since the database is generated when deploying this web application. After doing this you should have installed a [PostgreSQL](https://www.postgresql.org/) database instance. In the config.yaml file you must indicate the access configuration to this database.

In addition, it is necessary to install a library to manage the database from Python:

```bash
$ sudo apt install libpq-dev
```

### Backend installation

It is recommended that you use a separate Python 3.8 virtual environment for this application.

The required packages can be installed as follows:

```bash
$ pip install -r requirements.txt
```

Now we can download the source code and some additional data needed:

```bash
$ git clone https://github.com/jrmtnez/hnfc-agent
$ cd hnfc-agent
$ python install/install_nlp_resources.py

```

## Dataset

The dataset, with the access URLs to the original articles and the annotations made at the sentence level, is available at the following link: https://doi.org/10.5281/zenodo.10802196


## License and Disclaimer
Please see the LICENSE file for details. Downloading data indicates your acceptance of our disclaimer.

## Citation
```bibtex
@article{martinez-rico_building_2024,
	title = {Building a framework for fake news detection in the health domain},
	language = {en},
	author = {Martinez-Rico, Juan R and Araujo, Lourdes and Martinez-Romo, Juan},
	year = {2024}
}
```



