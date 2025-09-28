# Machine Learning Notebooks

This project aims at teaching you the fundamentals of Machine Learning in python. It contains the example code and solutions to the exercises in the Machine Learning courses purposed by SenIA.


# Quick Start

Want to play with these notebooks online without having to install anything?
Use any of the following services (I recommended Colab or Kaggle, since they offer free GPUs and TPUs).

WARNING: Please be aware that these services provide temporary environments: anything you do will be deleted after a while, so make sure you download any data you care about.

- Open In Colab
- Open in Kaggle

Want to install this project on your own machine?
Start by installing [Anaconda](https://www.anaconda.com/download) (or [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)), [git](https://git-scm.com/downloads)
Next, clone this project by opening a terminal and typing the following commands (do not type the first $ signs on each line, they just indicate that these are terminal commands):

```
$ git clone https://github.com/MouslyDiaw/handson-machine-learning.git
$ cd handson-machine-learning
```

Next, run the following commands:
```
$ conda env create -f environment.yml
$ conda activate handson
$ python -m ipykernel install --user --name=handson
```

Finally, start Jupyter:

```
$ jupyter lab
```


# FAQ
Which Python version should I use?
I recommend Python >= 3.12