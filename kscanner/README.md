#### KScanner
<u>KScanner</u>: is a novel combination of DBSCAN and K-Means clustering that uses automated Epsilon and MinPts approximation to identify the correct model parameters for DBSCAN. The output of DBSCAN is then fed to K-Means as the value for n_clusters to lower the total amount of iterations to determine appropriate cluster amounts.


#### How to use
from kscanner.scanners import kscan
kscan(data, graph=True, kmeans_n_init=100, kmeans_max_iter=1000, kmeans_tol=0.0001)


#### DBSCAN
<u>Density-Based Spatial Clustering of Applications with Noise (DBSCAN)</u>: is a density-based clustering algorithm that separates the high-density regions of the data from the low density regions. DBSCAN groups data points by distance, usually Euclidean, and the minimum number of points. Unlike K-Means clustering DBSCAN is not sensitive to outliers as they show up in low-density regions.
* DBSCAN Parameters
    * <u>Epsilon (EPS)</u>: This is the main threshold used for DBSCAN and is the minimum distance apart required for two points to be classified as neighbors.
    * To calculate the value of Eps, we take the distance between each data point to its closest neighbor using Nearest Neighbours
    * Then we can sort and plot them. From the plot, we identify Epsilon as the maximum value at the curvature of the graph
    * <u>MinPoints</u>: This parameter is the threshold for the minimum number of points needed to construct a cluster. Something is only a cluster in DBSCAN if the number of points in it is greater than or equal to MinPoints. Importantly, the point itself is included in the calculation.
        * Selecting MinPoints
            * If the dataset has 2 dimensions, use 4
            * If the dataset has > 2 dimensions, choose MinPts = 2*dim (Sander et al., 1998).
            * For larger datasets, with much noise, it suggested to go with minPts = 2 * D.
                * <u>Distance metric</u>: The distance metric used when calculating distance between instances of a feature array (typically Euclidean distance)
* Post clustering we are left with 3 types of data
    * <u>Core</u>: A point which is equal or greater than MinPoints and is within the Eps distance
    * <u>Border</u>: A point which has at least one Core point within Eps distance from itself
    * <u>Noise</u>: a point less than MinPoints within desitance Eps from itself


#### K-Means
<u>K-Means Clustering</u>: K-means clustering is a popular unsupervised machine learning algorithm. K-means identifies "k" number of centroids and then allocates every data point to the nearest cluster, while keeping the centroids as small as possible. The "means" refers to averaging the data points (i.e. finding the centroid).


File Descriptions:
* package_name: represents the main package.
* docs: includes documentation files on how to use the package.
* scripts: your top-level scripts.
* src: where your code goes. It contains packages, modules, sub-packages, and so on.
* tests: where you can put unit tests.
* LICENSE.txt: contains the text of the license (for example, MIT).
* CHANGES.txt: reports the changes of each release.
* MANIFEST.in: where you put instructions on what extra files you want to include (non-code files).
* README.txt: contains the package description (markdown format).
* pyproject.toml: to register your build tools.
* setup.py: contains the build script for your build tools.
* setup.cfg: the configuration file of your build tools.
