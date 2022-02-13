## Project 2

Anne Louise Seekford

### Overview

For project two, I applied two regression techniques, **Locally Weighted Linear (LOWESS) and Random Forest**, on the Boston Housing Dataset. The Boston Housing Dataset is derived from information collected by the U.S. Census Service concerning homes in the area of Boston, Massachusetts. Although the dataset contains over fifteen columns, each a distinct characteristic of a home in Boston, I will be focusing on only two. The input feature will be 'rooms', describing the average number of rooms per dwelling. The target, which will be predicted, is 'cmedv', the median value of owener-occupied homes (in thousands of dollars). 

Snippet of Dataset: 
<img width="1272" alt="Screen Shot 2022-02-12 at 9 25 52 PM" src="https://user-images.githubusercontent.com/71660299/153735617-137a6461-dc26-4c5e-b3de-52f41bd64b56.png">

Descriptions of my process, code, and methods used will be described as follows:

### Locally Weighted Linear Regression (LOWESS)

Stemming from Linear Regression, LOWESS can be considered a non-parametric algorithm that must use all the dataset for estimation (Figueira, 2021). 


Equation for Locally Weighted Linear Regression: 

<p align = 'center'> <img width="306" alt="Screen Shot 2022-02-12 at 11 19 23 PM" src="https://user-images.githubusercontent.com/71660299/153738433-1c60e39b-c996-48df-8b3d-4cea7e03a88f.png">

- yhat is obtained as a different linear combination of the values of y



##### How did we get this equation from a simple linear equation?

First, linear regression - the assumption that: 

      <img width="171" alt="Screen Shot 2022-02-12 at 11 15 56 PM" src="https://user-images.githubusercontent.com/71660299/153738371-656b4681-e83f-4daf-8322-3f9646b9b8d3.png">

So, if we pre-multiply this equation with a **matrix** of weights we get: <img width="333" alt="Screen Shot 2022-02-12 at 11 19 48 PM" src="https://user-images.githubusercontent.com/71660299/153738443-5b0e3cc1-36f2-478e-a0f5-5302de978eef.png">. Keep in mind here that *the "weights" are on the main diagonal and the rest of the elements are 0*. 

- The independent observations are the rows of the matrix *X* 
- Each row has a given number of columns (*number of features*), denoted by *p*. 
- Thus, every row is a vector in R^*p*. 
- The distance between two independent observations is the **Euclidean distance** between the two represented $p$-dimensional vectors. Euclidean distance is also commonly referred to as *L2 Norm*. 

As a result, this equation is as follows: 
<img width="471" alt="Screen Shot 2022-02-12 at 11 21 10 PM" src="https://user-images.githubusercontent.com/71660299/153738473-32a25afa-f337-46b9-befd-6945e3f18e4e.png">
- We shall have $n$ differenct weight vectors because we have $n$ different observations. 


**Linear regression can be seen as a linear combination of the observed outputs, or values of the target.**

  - To get to LOWESS, we have: 
       - <img width="202" alt="Screen Shot 2022-02-12 at 11 21 35 PM" src="https://user-images.githubusercontent.com/71660299/153738477-c853bb25-d0ae-46ff-8fd6-d16012620541.png">
  - We solve for *beta* (by assuming that *X^TX* is invertible): 
  <img width="415" alt="Screen Shot 2022-02-12 at 11 22 31 PM" src="https://user-images.githubusercontent.com/71660299/153738494-34e34fe8-6d8e-4bf8-836c-36cc95c3e7aa.png">
  - We take the expected value of this equation and obtain: 
 <img width="222" alt="Screen Shot 2022-02-12 at 11 22 55 PM" src="https://user-images.githubusercontent.com/71660299/153738503-898e6df6-59bf-4b19-a30e-dc46c0ae5f33.png">
  - Therefore, the predictions we make are: <img width="97" alt="Screen Shot 2022-02-12 at 11 23 32 PM" src="https://user-images.githubusercontent.com/71660299/153738521-a46d9795-d3ce-4b79-802d-408f5381a789.png">


For the locally weighted regression we have:
**$$  \large \hat{y} = X (X^{T}WX)^{-1}(X^{T}Wy) $$**

```
**In Locally Weighted Linear Regression, the predictions made are a linear combination of the actual observed values of the dependent variable.**




* Math equations

```markdown

`Code` text

```

### Random Forest

* Describe Decision trees*
* Describe RF
* Math equations

```markdown
Code

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

Figueira, J. P. (2021, June 1). Loess. Medium. Retrieved February 13, 2022, from https://towardsdatascience.com/loess-373d43b03564 

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/alseekford/410_Project2/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
