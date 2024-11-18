# DeepLearningTheory
Git repository containing the generating code for a jupyter-book containing various notes and calculations on Deep Learning Theory, PhD in physics at UniPR. 

The book is available online at website adress `https://vincenzozimb.github.io/DeepLearningTheory/` 

---

(If it does not already exist) Create the conda environment for this project, typing in the terminal: 

`conda env create -f env.yml`

Activate the environment with:

`conda activate phd-book`

To update the environment file (do it every time a new package is istalled), type:

`conda env export > env.yml`

---

To build the book type in the same folder of this README file:

`jupyter-book build book/`

Static link of the book (local, umpublished):

`file:///home/vincenzo/Desktop/PhD/DeepLearningTheory/book/_build/html/index.html`


---

To update the online website, rebuild the jupyter-book and type `ghp-import -n -p -f book/_build/html`  to push the newly built HTML to the `gh-pages` branch.