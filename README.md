# Pokemon Recognition

This repository is some pactice of data parsing by using Python. Notice that this repository is lab in "Workshop on AI & Big Data Analytics 2018".

---
## Prerequisite

* Before executing, you need to install the following packages
    * Install **Pillow 2.2.1** using `pip`
        ```bash
        $ [sudo] pip install Pillow==2.2.1
        ```
    * Install **bunch 1.0.1** using `pip`
        ```bash
        $ [sudo] pip install bunch
        ```
    * Install **scikit-learn 0.19.1** using `pip`
        ```bash
        $ [sudo] pip install scikit-learn
        ```

---
## Execution 

```bash
# Execute each main.py by using SVM training model
$ python main.py svm
$ python main.py SVM

# Execute each main.py by using KNN training model
$ python main.py knn
$ python main.py KNN
```

---
## Framework

* `main.py` - Use Python with respect to KNN and SVM training model to recognize each Pokemon
* `pokemon/` - Input image of each pokemon
* `pokemon_processed/` - Processed image after transfer into grayscale

---
## References

* [Pillow 2.2.1](https://pypi.org/project/Pillow/2.2.1/)
* [bunch 1.0.1](https://pypi.org/project/bunch/)
* [scikit-learn 0.19.1](https://pypi.org/project/scikit-learn/)

---
## Author

* [David Lu](https://github.com/yungshenglu)
