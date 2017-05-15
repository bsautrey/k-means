Written by Ben Autrey: https://github.com/bsautrey

---Overview---

Implement k-means clustering from Andrew Ng's CS229 course: http://cs229.stanford.edu/notes/cs229-notes7a.pdf.

tol - Stopping criteria. In particular, if the locations of kmeans no longer changes by tol, stop.
kmeans - the means.
memberships - What cluster does each x in X belong to?

---Requirements---

* numpy: https://docs.scipy.org/doc/numpy/user/install.html
* matplotlib: https://matplotlib.org/users/installing.html

---Example---

1) Change dir to where kmeans.py is.

2) Run this in a python terminal:

from kmeans import KMeans
km = KMeans()
km.generate_example()

OR

See the function generate_example() in kmeans.py.