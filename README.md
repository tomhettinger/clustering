# Clustering with KMeans and AgglomerativeClustering

`pandas` `sklearn` `numpy` `seaborn` `matplotlib`

This project is an exercise in unsupervised machine learning.  In this exercise, I generate a synthetic data set describing the physical characteristics of a population of 200 individuals.  The values of the characteristics are drawn from distributions in a way, such that there are sub-groups of individuals having similar values among some characteristics (e.g. Short-Active individuals have below average height, weight, and heart rate).  After data synthesis, K-Means clustering models were fit using various cluster count parameters, in an attempt to recover the original clustering characteristics.  Afterwards a hierarchical clustering approach is used to identify clustering behavior.


## Data Synthesis

The synthetic data set will consist of 200 individuals with the following characteristics, and default population parameters:

* Age in years (mu=45, std=5)
* Weight in kg (mu=60, std=3)
* Heart rate in bpm (mu=80, std=10)
* Height in cm (mu=160, std=10 for females), (mu=170, std=10 for males)
* Gender (1:1 female-to-male)

Using these baseline distribution parameters, individuals were synthesized in groups, where some of the characteristic distributions were modified.  The following 5 distributions comprise the entire population:

### Short-Active

Height, weight, and heart rate below average.

* Weight (mu=50, std=3)
* Heart rate (mu=45, std=10
* Height (mu=130 & 140, std=10)

### Tall-Active

Weight and heart rate below average, but height above average.

* Weight (mu=49, std=3)
* Heart rate (mu=50, std=10
* Height (mu=190 & 200, std=10)

### Docile-Male

Weight and heart rate above average.  All males.

* Weight (mu=70, std=3)
* Heart rate (mu=116, std=10)
* Female fraction = 0.0

### Tall-Slender

Height above average, weight below average.  Average heart rate.

* Weight (mu=51, std=3)
* Height (mu=188 & 197, std=10)


### Heavy-Active-Female

Weight above average and heart rate below average.  All female.

* Weight (mu=12+60, std=3)
* Heart rate (mu=43, std=10)
* Female fraction = 1.0



Note: that there exists significant overlap in characteristics among various groups.  For example, Short-Active individuals and Tall-Active individuals have similar weights and heart rates, but differ in height.  Also notice that Age is not a discriminating factor for any of the groups.

The total population size is 200 with group sizes: 50 from Short-Active, 70 from Tall-Active, 35 from Docile-Male, 35 from Tall-Slender, and 10 from Heavy-Active-Female.

Here is a random sampling of 15 rows from the population (data.csv).


|  id  |      group            | age (years)  | weight (kg)  | heartrate (bpm) |  height (cm) | gender |
|------|-----------------------|--------------|--------------|-----------------|--------------|--------|
| 162  |  Tall-Slender         |  39          |   54         |     73          |   209        |    m   |
| 120  |  Docile-Male          |  39          |   72         |    123          |   156        |    m   |
| 88   |  Tall-Active          |  41          |   48         |     52          |   185        |    f   |
| 6    |  Short-Active         |  40          |   48         |     43          |   136        |    m   |
| 84   |  Tall-Active          |  41          |   48         |     37          |   209        |    m   |
| 145  |  Docile-Male          |  41          |   63         |    127          |   153        |    m   |
| 129  |  Docile-Male          |  44          |   62         |    114          |   154        |    m   |
| 191  |  Heavy-Active-Female  |  45          |   76         |     29          |   170        |    f   |
| 151  |  Docile-Male          |  44          |   67         |    112          |   185        |    m   |
| 53   |  Tall-Active          |  45          |   52         |     70          |   196        |    f   |
| 93   |  Tall-Active          |  43          |   46         |     56          |   196        |    f   |
| 30   |  Short-Active         |  52          |   54         |     46          |   149        |    m   |
| 76   |  Tall-Active          |  43          |   46         |     50          |   181        |    f   |
| 51   |  Tall-Active          |  39          |   52         |     64          |   184        |    m   |
| 3    |  Short-Active         |  36          |   47         |     50          |   146        |    f   |


Using `seaborn`, I've created some figures showing distributions of values in the data set.

![violin](figures/violin.png)

![swarm](figures/swarmplot.png)


In terms of correlation, only weight and heart rate are strongly correlated, due to Active individuals and Docile individuals.

![heatmap](figures/heatmap.png)


The following pairwise plot shows feature-feature distributions, with color-coding based on group membership.

![pairwise](figures/pairplot.png)


Before beginning the machine learning exercise, all group memberships are hidden.  The pairwise plot without group color-coding is presented here.

![pairwise_nocolor](figures/pairplot_kde.png)


## KMeans

Using the `KMeans` classifier in the `sklearn.cluster` module, I processed the data set (without labels) to cluster the data into k clusters.  The number of clusters, k, varied from 2 to 10.  For each cluster count k, the KMeans algorithm ran a total of 10 times with new random starting points for cluster centroids.  The best fit (minimizing square distances from cluster centroids) of the 10 trials is ultimately adopted for the classifications given at that k.  After running k=2,3,...,10 models with 10 iterations, the algorithm yields a resulting k=2,3,...,10 sets of classification predictions.

The following pairwise figures depict the distributions of parameters, once again, but with color-coding given by the class applied to each observation.  Shown below are the classifications given from the models k=2 through k=6.

![pairwise_k2](figures/pairplot_k2_class.png)
![pairwise_k3](figures/pairplot_k3_class.png)
![pairwise_k4](figures/pairplot_k4_class.png)
![pairwise_k5](figures/pairplot_k5_class.png)
![pairwise_k6](figures/pairplot_k6_class.png)

How well does the clustering do?  We can take a look at confusion matricies (below) and a few figures comparing the original grouping with the model clustering.  For example, we can look at height vs heartrate for the original grouping (stars) and the model-generated clusters (circles):

<img src="figures/height_v_heartrate_true.png" height="500">
<img src="figures/height_v_heartrate_pred.png" height="500">

There seems to be strong agreement when comparing Docile-Male group with group 2, but there is confusion when differentiating Tall-Active and Tall-Slender (despite the difference in heart rates).  Below we show, similarly, the same comparison looking at height vs weight.

<img src="figures/height_v_weight_true.png" height="500">
<img src="figures/height_v_weight_pred.png" height="500">

Alternatively, we can look at a confusion matrix, if we assume labeling for the K-Means clustering.  For example in the k=2 model, if we assume that group 1 == "Docile Males", then the confusion matrix would be:

|             | Docile-Male | other |
|-------------|-------------|-------|
| Docile-Male | 35          |   0   |
| Other       | 1           |  164  |

The k=2 model does a great job at separating the Docile-Male group from the others (this is the only all-male group). Likewise, for the k=3 model:

| Tall-Active or Tall-Slender | Docile-Male | Heavy-Active-Female or Short-Active |
|-----------------------------|-------------|-------------------------------------|
|             104             |      0      |               1                     |
|              0              |     35      |               0                     |
|              0              |      0      |              60                     |

Here, the k=3 model separates out the Docile-Male group, and splits the remaining individuals into two clusters, each containing almost exclusively different groups. For k=5, three of the five groups are clustered into mostly homogeneous clusters, but two groups (Tall-Active and Tall-Slender) are not separated, despite their differences in heart rate.

| Tall-Slender | Tall-Active | Docile-Male | Short-Active | Heavy-Active-Female |
|--------------|-------------|-------------|--------------|---------------------|
|     20       |    15       |    0        |    0         |         0           |    
|     33       |    37       |    0        |    0         |         0           |    
|      0       |    0        |    35       |    0         |         0           |    
|      0       |    0        |    0        |   30         |        20           |    
|      0       |    0        |    0        |    0         |        10           |    


In practice, the group classifications will be unknown.  This means we need a way to decide what level of clustering (how many clusters) is appropriate.  One way to accomplish this is by looking at the decrease in inertia (square of distances to cluster centroids), and finding a elbow where the increase in cluster count k no longer reduces the inertia sufficiently.  Below, I've plotted the inertia of models as a function of k, as well as the derivative of the inertia.

![inertia](figures/inertia.png)
![inertia_derivative](figures/inertia_derivative.png)

For k values > 5 or 6, the decrease in inertia is not very large.



## AgglomerativeClustering

Text Text Text.
