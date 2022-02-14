## Project 2

Anne Louise Seekford

### Overview

For project two, I applied two regression techniques, **Locally Weighted Linear (LOWESS) and Random Forest**, on the Boston Housing Dataset. The Boston Housing Dataset is derived from information collected by the U.S. Census Service concerning homes in the area of Boston, Massachusetts. Although the dataset contains over fifteen columns, each a distinct characteristic of a home in Boston, I will be focusing on only two. The input feature will be 'rooms', describing the average number of rooms per dwelling. The target, which will be predicted, is 'cmedv', the median value of owener-occupied homes (in thousands of dollars). 

Snippet of Dataset: 
<img width="1272" alt="Screen Shot 2022-02-12 at 9 25 52 PM" src="https://user-images.githubusercontent.com/71660299/153735617-137a6461-dc26-4c5e-b3de-52f41bd64b56.png">

Descriptions of my process, code, and methods used will be described as follows:

### Locally Weighted Linear Regression (LOWESS)

Stemming from Linear Regression, LOWESS can be considered a non-parametric algorithm that must use all the dataset for estimation (Figueira, 2021). 


Prediction Equation for Locally Weighted Linear Regression: 

<p align = 'center'> <img width="306" alt="Screen Shot 2022-02-12 at 11 19 23 PM" src="https://user-images.githubusercontent.com/71660299/153738433-1c60e39b-c996-48df-8b3d-4cea7e03a88f.png"> 

- *yhat* is obtained as a different linear combination of the values of y



##### How did we get this LOWESS prediction equation from a simple linear equation?

First, linear regression - the assumption that: 

  <p align = 'center'>    <img width="171" alt="Screen Shot 2022-02-12 at 11 15 56 PM" src="https://user-images.githubusercontent.com/71660299/153738371-656b4681-e83f-4daf-8322-3f9646b9b8d3.png">
    

So, if we pre-multiply this equation with a **matrix** of weights we get: 
    
       
<p align = 'center'> <img width="333" alt="Screen Shot 2022-02-12 at 11 19 48 PM" src="https://user-images.githubusercontent.com/71660299/153738443-5b0e3cc1-36f2-478e-a0f5-5302de978eef.png">. 
       
  
Keep in mind here that *the "weights" are on the main diagonal and the rest of the elements are 0*. 

- The independent observations are the rows of the matrix *X* 
- Each row has a given number of columns (*number of features*), denoted by *p*. 
- Thus, every row is a vector in R^*p*. 
- The distance between two independent observations is the **Euclidean distance** between the two represented $p$-dimensional vectors. Euclidean distance is also commonly referred to as *L2 Norm*. 

As a result, this equation is as follows: 
  
  
<p align = 'center'> <img width="471" alt="Screen Shot 2022-02-12 at 11 21 10 PM" src="https://user-images.githubusercontent.com/71660299/153738473-32a25afa-f337-46b9-befd-6945e3f18e4e.png">
  
  
- We shall have $n$ differenct weight vectors because we have $n$ different observations. 


**Linear regression can be seen as a linear combination of the observed outputs, or values of the target.**

  - To get to LOWESS, we have: 
  
     
<p align = 'center'> <img width="202" alt="Screen Shot 2022-02-12 at 11 21 35 PM" src="https://user-images.githubusercontent.com/71660299/153738477-c853bb25-d0ae-46ff-8fd6-d16012620541.png">
  
  
  - We solve for *beta* (by assuming that *X^TX* is invertible): 
  
 
<p align = 'center'> <img width="415" alt="Screen Shot 2022-02-12 at 11 22 31 PM" src="https://user-images.githubusercontent.com/71660299/153738494-34e34fe8-6d8e-4bf8-836c-36cc95c3e7aa.png">
  

  - We take the expected value of this equation and obtain: 
  
 
<p align = 'center'> <img width="222" alt="Screen Shot 2022-02-12 at 11 22 55 PM" src="https://user-images.githubusercontent.com/71660299/153738503-898e6df6-59bf-4b19-a30e-dc46c0ae5f33.png">
  

  - Therefore, the predictions we make are: 
  
  
 <p align = 'center'> <img width="97" alt="Screen Shot 2022-02-12 at 11 23 32 PM" src="https://user-images.githubusercontent.com/71660299/153738521-a46d9795-d3ce-4b79-802d-408f5381a789.png">
   


Finally, that takes us to the locally weighted regression we have:
   
 
<p align = 'center'> <img width="306" alt="Screen Shot 2022-02-12 at 11 19 23 PM" src="https://user-images.githubusercontent.com/71660299/153738433-1c60e39b-c996-48df-8b3d-4cea7e03a88f.png">
  

   

**In Locally Weighted Linear Regression, the predictions made are a linear combination of the actual observed values of the dependent variable.**



##### Code and Methods

  This Lowess Regressor was used to "fit" the data:
```markdown

  `def lowess_reg(x, y, xnew, kern, tau):
      # tau is called bandwidth K((x-x[i])/(2*tau))
      # We expect x to the sorted increasingly
      n = len(x)
      yest = np.zeros(n)

    #Initializing all weights from the bell shape kernel function    
    w = np.array([kern((x - x[i])/(2*tau)) for i in range(n)])     
    
    #Looping through all x-points
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        #theta = linalg.solve(A, b) # A*theta = b
        theta, res, rnk, s = linalg.lstsq(A, b)
        yest[i] = theta[0] + theta[1] * x[i] 
    f = interp1d(x, yest,fill_value='extrapolate')
    return f(xnew)`

```
  
  
Additionally, I ran the regressor on three distinct kernels: Tricubic, Epanechnikov, and Quartic
  - Each kernel was specified in the "kern" parameter in the lowess_reg function
  
```markdown

  `def tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)`
  
  `def Epanechnikov(x):
  return np.where(np.abs(x)>1,0,3/4*(1-np.abs(x)**2))`
  
  `def Quartic(x):
  return np.where(np.abs(x)>1,0,15/16*(1-np.abs(x)**2)**2)`
  
```
  
I estimated a value for the LOWESS regressor using each kernel and k-fold cross validation 
  with 10 splits.
  To compare results, I calculated the mean squared error of each, which are as follows: 
  
```markdown


  
  - LOWESS Tricubic 10-Fold Cross Validated MSE = $36,384.16
  - LOWESS Epanechnikov 10-Fold Cross Validated MSE = $36,453.06
  - LOWESS Quartic 10-Fold Cross Validated MSE = $36,341.57
  

```
  

### Random Forest

  
##### Decision Trees
  
  In a similar way to how LOWESS stems from linear regression, random forests are built upon decision trees. 
  Decision trees can be used to solve classification and regression problems, consisting of a top-down structure where each internal node represents a "test" on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label (Liberman, 2020).  
  
  <p align = 'center'> <img width="627" alt="Screen Shot 2022-02-13 at 4 08 30 PM" src="https://user-images.githubusercontent.com/71660299/153775123-7eec698a-2374-4325-87c6-9c39efdd6fe7.png">
    
 <p align = 'center'>   <img width="517" alt="Screen Shot 2022-02-13 at 4 09 02 PM" src="https://user-images.githubusercontent.com/71660299/153775134-af560130-817a-4c04-bafd-b827cbcead8f.png"> (Javapoint)


  
##### Random Forests
  
  In order to minimize bias and error (due to variance) in decision trees, we can introduce a random forest. Simply put, a random forest is a series of decision trees. 
  To reduce variance, random forest (as applied in this Boston Housing example) seperates the data into training and testing samples. In this case, 25% of the data went to test, while the remaining 75% was trained. An additional benefit of random forest is it prevents overfitting. 
   
   
Here is an example of the structure: 
   
 <p align = 'center'>  <img width="665" alt="Screen Shot 2022-02-13 at 4 12 20 PM" src="https://user-images.githubusercontent.com/71660299/153775240-695a0dc8-95b6-434c-a005-c634c2ec240f.png"> (Javapoint)


  
For Random Forest, I fit on the scaled data and plotted the results: 
  
  
```markdown
 `rf_model = RandomForestRegressor(n_estimators=150, max_depth=3)`
 `rf_model.fit(xtrain_scaled, ytrain)`
 `rf_yhat = rf_model.predict(xtest_scaled)`
```  
  
  
 <p align = 'center'> <img width="631" alt="Screen Shot 2022-02-13 at 3 35 20 PM" src="https://user-images.githubusercontent.com/71660299/153773840-a73fac7e-06a9-45e9-8ed4-7181a1d04455.png">
   
   
In order to improve upon this plot, it was necessary to sort the matrix by x values.
   By doing so, the y values will move accordingly and the plot improved signficantly. 
   
   
   <p align = 'center'>  <img width="663" alt="Screen Shot 2022-02-13 at 3 36 41 PM" src="https://user-images.githubusercontent.com/71660299/153773892-c1934e46-a971-4988-9623-a606d420c438.png">
     


To compare the results of the Random Forest Regressor with LOWESS, the mean squared error was 
     calculated using 10-fold cross validation. 
     
     
  
```markdown
Random Forest 10-Fold Cross Validated MSE = $33,366.94
```


### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).
     
     

### References

Liberman, N. (2020, May 21). Decision trees and random forests. Medium. Retrieved February 13, 2022, from https://towardsdatascience.com/decision-trees-and-random-forests-df0c3123f991 
     
Figueira, J. P. (2021, June 1). Loess. Medium. Retrieved February 13, 2022, from https://towardsdatascience.com/loess-373d43b03564 
     
Machine learning decision tree classification algorithm - javatpoint. www.javatpoint.com. (n.d.). Retrieved February 13, 2022, from https://www.javatpoint.com/machine-learning-decision-tree-classification-algorithm 
     
Machine learning random forest algorithm - javatpoint. www.javatpoint.com. (n.d.). Retrieved February 13, 2022, from https://www.javatpoint.com/machine-learning-random-forest-algorithm 
     

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/alseekford/410_Project2/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.
     
     

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
