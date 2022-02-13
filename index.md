## Project 2

Anne Louise Seekford

### Overview

For project two, I applied two regression techniques, **Locally Weighted Linear (LOWESS) and Random Forest**, on the Boston Housing Dataset. The Boston Housing Dataset is derived from information collected by the U.S. Census Service concerning homes in the area of Boston, Massachusetts. Although the dataset contains over fifteen columns, each a distinct characteristic of a home in Boston, I will be focusing on only two. The input feature will be 'rooms', describing the average number of rooms per dwelling. The target, which will be predicted, is 'cmedv', the median value of owener-occupied homes (in thousands of dollars). 

Snippet of Dataset: 
<img width="1272" alt="Screen Shot 2022-02-12 at 9 25 52 PM" src="https://user-images.githubusercontent.com/71660299/153735617-137a6461-dc26-4c5e-b3de-52f41bd64b56.png">

Descriptions of my process, code, and methods used will be described as follows:

### Locally Weighted Linear Regression (LOWESS)

Stemming from Linear Regression, LOWESS can be considered a non-parametric algorithm that must use all the dataset for estimation (Figueira, 2021). 

* Math equations


```markdown
Equation for Locally Weighted Linear Regression:
$$
\large \hat{y} = X (X^{T}WX)^{-1}(X^{T}Wy)
$$

- $ \hat{y}$ is obtained as a different linear combination of the values of y

```

How did we get this equation from a simple linear equation?
```markdown

First, linear regression - the assumption that: $$\ y = X\cdot\beta +\sigma\epsilon $$. 

So, if we pre-multiply this equation with a **matrix** of weights we get: $$\ W(i)y = W(i)X\cdot\beta +\sigma W(i)\epsilon $$. Keep in mind here that *the "weights" are on the main diagonal and the rest of the elements are 0*. 

- The independent observations are the rows of the matrix $X$ 
- Each row has a given number of columns (*number of features*), denoted by $p$. 
- Thus, every row is a vector in $\mathbb{R}^p$. 
- The distance between two independent observations is the **Euclidean distance** between the two represented $p$-dimensional vectors. Euclidean distance is also commonly referred to as *L2 Norm*. 

As a result, this equation is as follows: $$ dist(\vec{v}, \vec{w}) = \sqrt{(v_1 - w_1)^2 + (v_2 - w_2)^2 + ... + (v_p - w_p)^2}$$
- We shall have $n$ differenct weight vectors because we have $n$ different observations. 


**Linear regression can be seen as a linear combination of the observed outputs, or values of the target.**

  - To get to LOWESS, we have: $$ X^Ty = X^TX\beta + \sigma X ^T\epsilon $$
  - We solve for $\beta$ (by assuming that $X^TX$ is invertible): 
  $$ \large \beta = (X^TX)^{-1}(X^Ty) - \sigma (X^TX)^{-1} X^T \epsilon $$
  - We take the expected value of this equation and obtain: 
  $$ \large \hat\beta = (X^TX)^{-1}(X^Ty) $$
  - Therefore, the predictions we make are: $$ \large \hat{y} = X \hat\beta $$

For the locally weighted regression we have:
**$$  \large \hat{y} = X (X^{T}WX)^{-1}(X^{T}Wy) $$**


  - For LOWESS, $ \hat{y}$ is obtained as a different linear combination of the values of y.

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
