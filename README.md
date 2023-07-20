# Deep Learning for Efficient and Optimal Motion Planning for AUVs with Disturbances

## Introduction

Code used to obtain the results in the paper Parras, J., Apellániz, P.A., & Zazo, S. (2021). Deep Learning for Efficient and Optimal Motion Planning for AUVs with Disturbances. Sensors 21(15), 5011. [DOI](https://doi.org/10.3390/s21155011).

## Launch

This project has been tested on Python 3.7.0 on Ubuntu 20.04. To run this project, create a `virtualenv` (recomended) and then install the requirements as:

```
$ pip install -r requirements.txt
```

To show the results obtained in the paper, simply run the main file as:
```
$ python main_dgm.py
```

In case that you want to train and/or test, set the train and/or test flag to `True` in the `main_dgm.py` file and then run the same order as before. Note that the results file will be overwritten. 
