# Principal-Component-Analysis
Project completed in pursuit of Master's of Science in Data Analytics.

## PART I: RESEARCH QUESTION

### PROPOSAL OF QUESTION

Can Principal Component Analysis (PCA) be used to identify which patient metrics contribute most to the variability in the dataset while significantly reducing the dimensions of the dataset? 

### DEFINED GOAL

The primary goal of this analysis is to reduce the dimensions of the dataset significantly while still maintaining the as much of the original information as possible (IBM, 2023). 

## PART II: METHOD JUSTIFICATION

### EXPLANATION OF PCA

Before one can begin the PCA, you have to remove all categorical variables, as PCA only works on numeric data. Next, you need to only keep the continuous variables by removing any discrete ones. In the medical dataset, this only leaves nine variables: Lat, Lng, Age, Income, VitD_levels, vitD_supp, Initial_days, TotalCharge, and Additional_charges. 

Next, the remaining variables must be scaled for the PCA to work most effectively. All outliers should be treated, and any missing or null values treated as well. In this dataset, there were none to clean. Once the data is then scaled, you can perform PCA. 

PCA analyzes the dataset by then transforming the high-dimensional dataset into a lower-dimensional one by identifying the directions of maximum variance, which are called principal components. The first principal component (PC1) explains the maximum amount of variance in the original features. The next principal component (PC2) explains the maximum variance in what is left after PC1. PCA will analyze all of the nine variables in this dataset, so there will be 9 different principal components each which less explained variance than the previous one.

Looking at the nine variables, I believe that several of them will be highly correlated, and therefore can be reduced by PCA. For example, I believe Lat and Lng will show a high correlation. Also, VitD_levels and vitD_supp also will be highly correlated. Both ‘charge’ variables should be highly correlated as well. Initial_days has a direct correlation to hospital fees, so I believe this will also be highly correlated. Therefore, I expect logically that the dimensions of this dataset should be shown to be able to be reduced down to five principal components using PCA. 

### PCA ASSUMPTION

One assumption to PCA is that there are highly correlated variables in the dataset (Santos, 2024).

## PART III: DATA PREPARATION

### CONTINUOUS DATA SET VARIABLES

There are nine continuous variables in the medical dataset. They are as follows:
- Lat
- Lng
- Age
- Income
- VitD_levels
- vitD_supp
- Initial_days
- TotalCharge
- Additional_charges

## PART IV: ANALYSIS

### PRINCIPAL COMPONENTS

![IMG_1638](https://github.com/user-attachments/assets/71226e72-5653-486a-9c57-a1764c1e7b25)

### IDENTIFICATION OF THE TOTAL NUMBER OF COMPONENTS

The total number of principal components is best calculated using the Kaiser Criterion method. The Kaiser Criterion method analyzes which factors in the dataset explain the most proportion of variance. Choosing the right number of factors is best done by finding the eigenvalues of each factor, and then retaining only those where the eigenvalues are greater than 1. Under these rules, the best choice in this analysis is then four principal components. However, PC5’s eigenvalue is 0.998, and PC6’s is 0.987 which are both extremely close to 1. Using an eigenvalue of greater than 1 is a general rule, in that eigenvalues less than 1 explain less variance, but in this dataset, these values are also extremely close and therefore do explain a good amount of variance. For these reasons, I will keep six total principal components instead of four. 

![IMG_1639](https://github.com/user-attachments/assets/09805144-623a-4602-b1b4-730bf09281cd)

### VARIANCE OF EACH COMPONENT
```python
exp_var = pca.explained_variance_ratio_

print(f"The variance of the first   principal component (PC1) is: {exp_var[0] * 100:.2f}%")
print(f"The variance of the second  principal component (PC2) is: {exp_var[1] * 100:.2f}%")
print(f"The variance of the third   principal component (PC3) is: {exp_var[2] * 100:.2f}%")
print(f"The variance of the fourth  principal component (PC4) is: {exp_var[3] * 100:.2f}%")
print(f"The variance of the fifth   principal component (PC5) is: {exp_var[4] * 100:.2f}%")
print(f"The variance of the sixth   principal component (PC6) is: {exp_var[5] * 100:.2f}%")
```
> The variance of the first   principal component (PC1) is: 22.15%
>
> The variance of the second  principal component (PC2) is: 19.04%
>
> The variance of the third   principal component (PC3) is: 12.37%
>
> The variance of the fourth  principal component (PC4) is: 11.28%
>
> The variance of the fifth   principal component (PC5) is: 11.09%
>
> The variance of the sixth   principal component (PC6) is: 10.97%

### TOTAL VARIANCE CAPTURED BY COMPONENTS
```python
total_variance_captured = np.sum(pca.explained_variance_ratio_[:6])

print(f"The total variance captured in all six principal components is: {total_variance_captured * 100:.2f}%")
```
> The total variance captured in all six principal components is: 86.90%

### SUMMARY OF DATA ANALYSIS

In this analysis, I first reduced the dataset from fifty variables down to twenty-three by removing all object datatypes. 
```python
df1 = df.select_dtypes(exclude = 'object')
df1.columns
```
![IMG_1640](https://github.com/user-attachments/assets/002c6ccd-9519-4450-8fc5-feb615774021)

Next, by reviewing these twenty-three variables, I was able to remove all except nine, leaving just the continuous variables. 
```python
#drop all discrete variables
cont_df = df1.drop(['CaseOrder', 'Zip', 'Children', 'Doc_visits', 'Full_meals_eaten', 'Population', 
                    'Item1', 'Item2', 'Item3', 'Item4', 'Item5', 'Item6', 'Item7', 'Item8'], axis=1)
```

I then scaled the resulting dataset using Scikit-Learn’s StandardScaler which normalized the data to prepare it for more efficient analysis in PCA. 
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

norm_df = scaler.fit_transform(cont_df)
```
I then performed the PCA function by fitting and transforming the scaled dataframe. 
```python
from sklearn.decomposition import PCA
pca = PCA()

PC = pca.fit_transform(scaled_df)
```
A loading matrix was created by aligning PC’s 1 through 9 in a matrix with all nine variables to show the contributions of the dataset variables to each of the principal components. 
```python
loading_matrix = pd.DataFrame(pca.components_, columns = cont_df.columns,
                             index = ('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9'))
```

To determine the correct number of principal components, I looked at scree plots using the Elbow Method and the Kaiser Criterion. The graph for the Elbow Method was inconclusive, making it hard to determine the correct number of principal components. 
```python
plt.figure(figsize=(7,5))
plt.plot(pcomp, exp_var, "b-")
plt.title('Scree Plot (Elbow Method)', fontsize=16)
plt.xlabel('Number of Component', fontsize=12)
plt.ylabel('Variance Proportion', fontsize=12)
plt.grid()
plt.show()
```
![IMG_1641](https://github.com/user-attachments/assets/654200d4-8237-4274-bbd3-c5790199ead0)

Based on this plot, the correct number of components is three. However, the total variance captured is only 53.56% which does not look like a good model. I then decided to plot the Kaiser Criterion. The Kaiser Rule uses eigenvalues of 1.0 and greater to determine the acceptance of a principal component. 
```python
#Kaiser Criterion

plt.figure(figsize=(7,5))
plt.plot(pcomp, var, 'b')
plt.title('Screen Plot (Kaiser Criterion)', fontsize=16)
plt.xlabel('Number of Component', fontsize=12)
plt.ylabel('Eigenvalues', fontsize=12)
plt.axhline(y=1, color='g', linestyle='dashdot')
plt.grid()
plt.show()
```
![IMG_1642](https://github.com/user-attachments/assets/a520d78a-10f0-4b46-ae13-84411c2325d6)

As earlier stated, the Kaiser method only has 4 components with eigenvalues larger than 1, but expanding this parameter slightly lower than 1 to 0.98 I now have 6 total components retained. Six components explain 86.9% of the variance of the original dataset. This is a good model, given such a high percentage of variance while also reducing the dimensions from fifty down to just six. 

## PART V: SUPPORTING DOCUMENTATION

#### SOURCES FOR THIRD-PARTY CODE

Scikit Learn (2007-2024). StandardScaler. Retrieved October 3, 2024, from https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html.

Scikit Learn (2007-2024). PCA. Retrieved October 8, 2024, from https://scikit-learn.org/dev/modules/generated/sklearn.decomposition.PCA.html.

Pandas (2023, June 28). Retrieved September 27, 2023, from https://pandas.pydata.org/docs/reference/index.html.

Waskom, M. (2012-2022). Seaborn Statistical Data Visualization. Retrieved September 27, 2023, from https://seaborn.pydata.org/index.html.

#### SOURCES 

Bruce, P.A. (2020). Practical statistics for data scientists. 50+ essential concepts using r and python. O’Reilly Media, Incorporated. WGU Library.

Larose, C.D., Larose, D.T. (2019) Data science using Python and R. Chichester, NJ: Wiley Blackwell.

IBM. (December 8, 2023). What is principal component analysis (PCA)? Retrieved October 8, 2024, from https://www.ibm.com/topics/principal-component-analysis.

Santos. J. (Aril 24, 2024). Principal Component Analysis (PCA). Retrieved October 8, 2024, from julius.ai/articles/principal-component-analysis-pca.
