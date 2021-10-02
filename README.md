# Learning log "Python machine leaning book 3rd edition"

see: https://github.com/rasbt/python-machine-learning-book-3rd-edition

# Start up
## 1st step
Install miniconda
- see: https://docs.conda.io/en/latest/miniconda.html

## 2nd step
Create new project from PyCharm

## 3rd step
Check conda env & install libraries from terminal
```console
$ conda create -n python_ml python=3.9
$ conda activate python_ml
$ conda env list
# conda environments:
#
base                     ~/opt/miniconda3
python_ml             *  ~/opt/miniconda3/envs/python_ml
$ conda install numpy scipy scikit-learn matplotlib pandas pydotplus nltk
$ brew install graphviz
```

## 4th step
Get dataset from keggle
- example: https://www.kaggle.com/datafiniti/pizza-restaurants-and-the-pizza-they-sell

## 5th step
Create notebooks from PyCharm

## 6th step
This project is using local package `pm3`.  
Add the package to interpreter Paths as follows

> - Project -> Python Interpreter  
> - Open interpreter select menu. And select `Shaow All`.  
> - Click icon. Tips is `Show paths for the selected interpreter`.
> - Add Path `<repository root>/notebooks`

## Enjoy! ML