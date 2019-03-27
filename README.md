# Pokemon Recognition

This repository is going to implement a Pokemon's image recognition. Besides, this program is using KNN and SVM model with **scikit-learn** to learn. Notice that this program can recognize the image of Pokemon as follow:
* Charizrd
* Gengar
* Pikachu
* Tangela

---
## Prerequisite

> **NOTICE:** Make sure you have already installed Python on your machine.

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

* Execute `main.py` by using **SVM** training model
    ```bash
    # Make sure your current directory is "src/"
    $ python main.py svm
    $ python main.py SVM
    ```
* Execute `main.py` by using **KNN** training model
    ```bash
    # Make sure your current directory is "src/"
    $ python main.py knn
    $ python main.py KNN
    ```

---
## File Description

* `./src/main.py` - Use Python with respect to KNN and SVM training model to recognize each Pokemon
* `./input/` - Input image of each pokemon
* `./processed/` - Processed image after transfer into grayscale

---
## References

* [Pillow 2.2.1](https://pypi.org/project/Pillow/2.2.1/)
* [bunch 1.0.1](https://pypi.org/project/bunch/)
* [scikit-learn 0.19.1](https://pypi.org/project/scikit-learn/)

---
## Contributor

> **NOTICE:** You can follow the contributing process [CONTRIBUTING.md](CONTRIBUTING.md) to join me. I am very welcome any issue!

* [David Lu](https://github.com/yungshenglu)

---
## License

[GNU GENERAL PUBLIC LICENSE Version 3](LICENSE)
