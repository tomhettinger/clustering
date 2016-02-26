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




## ?????
