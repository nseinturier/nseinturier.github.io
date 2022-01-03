---
layout: post
title: Simulate P-ball
---

# I. Introduction and Motivation



There is a strange result concerning the uniform distribution on the p-ball in high dimension :
<br>
Let's have 
<div> $$ X^{(p)} \sim \mathcal{U}(\mathcal{B_p(1)}) $$ </div>
<br>
with  
<div>$$ \mathcal{B_p(r)} = \{ x \in \mathbb{R^p} , \lVert x \rVert  \leq r \} \quad \lVert \rVert \text{ being the standard euclidean distance} $$ </div>
<br>
Then 
<br>
<div> $$ E(\lVert X^{(p)} \rVert^2) \underset{p \to +\infty}{\overset{}{\longrightarrow}}1$$ </div>

That is to say that in large dimensions, all the draws of a uniform distribution on the unit ball are concentrated on the edge. 
<br>
The Mathematical proof is at the end of this Notebook.

I found something interesting while trying to simulate this result. One of the first challenge is to uniformly sampling a p-ball. An elegant method for doing so ([Barthe, 2005](https://arxiv.org/abs/math/0503650)) is to randomly draw p coordinates $$X_1,…,X_p$$ i.i.d from a standard normal distribution. Also sample Y from the exponential distribution with parameter $$\lambda=1$$. Then the desired sample is :
<br>
<br>
<div> $$\frac{(X_1,...,X_p)}{\sqrt{Y+\sum_1^{p}X_i^2}}$$ </div>

However, and as mentionned [here](http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/), I found empirical evidence in this notebook that the right parameter for the exponential distribution should be $$\lambda = 1/2$$.

# II. Code


```python
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
```

## Functions


```python
# Create the desired number of sample of uniform random variable on the d-ball 

def random_nball(d = 1,sample = 1,exp_param = 2): # the exp_param is here to prove the point
    
    x = np.random.normal(0,1,d * sample) # normal
    x = x.reshape(sample,d) 
    e = np.random.exponential(exp_param,size = sample) # exponential
    e = e.reshape(sample,1)
    for_denom = np.concatenate((x**2,e),axis = 1) # concat with  x**2
    denom = (np.sum(for_denom,axis=1))**0.5
    
    return (x / denom[:,None])
```


```python
def monte_carlo(array):
    return (np.mean((LA.norm(array,axis = 1)**2)))
```

## Curve


```python
#calculate the expected value for each dimension from 1 to to 50 with monte-carlo algorithm with n = 10 000

dim = [i+1 for i in range(50)] 
results =[]
for i in dim:
    results.append(monte_carlo(random_nball(d = i,sample = 10000,exp_param = 1))) 
```


```python
#theorical results for the Expected value
th_results = [(i)/(i+2) for i in dim]
```


```python
plt.plot(dim,results,label = 'Monte-Carlo')
plt.plot(dim,th_results,label = "Theorical value")
plt.title('Expected Value vs Monte-Carlo for lambda = 1')
plt.ylabel('Expected value')
plt.xlabel('Dimension')
plt.legend(loc='lower right')
plt.show()
```


![png](/Notebooks/Ball/img/output_13_0.png)


The two curves do not seem to match. Let's plot the errors : 


```python
plt.plot(dim,np.abs(np.array(results)-np.array(th_results)))
plt.title('Errors ')
plt.ylabel('error in absolute value')
plt.xlabel('Dimension')
plt.show()
```


![png](/Notebooks/Ball/img/output_15_0.png)


## With parameter lambda = 1/2


```python
results =[]
for i in dim:
    results.append(monte_carlo(random_nball(d = i,sample = 10000,exp_param = 2))) 
# we use 2 for the parameter because numpy uses Beta = 1/lambda for exponential simulation
```


```python
plt.plot(dim,results,label='Monte-Carlo')
plt.plot(dim,th_results,label = "Theorical value")
plt.title( 'Expected Value vs Monte-Carlo for lambda = 1/2')
plt.ylabel('Expected value')
plt.xlabel('Dimension')
plt.legend(loc='lower right')
plt.show()
```


![png](/Notebooks/Ball/img/output_18_0.png)


a near perfect match


```python
plt.plot(dim,np.abs(np.array(results)-np.array(th_results)))
plt.title('Errors ')
plt.ylabel('error in absolute value')
plt.xlabel('Dimension')
plt.show()
plt.show()
```


![png](/Notebooks/Ball/img/output_20_0.png)


# Proof

first let's proove that 
<div>$$V_p(r) = r^pV_p(1)$$</div>
<br>
with 
<div> $$\begin{align}
V_p(r) &= \text{Volume}(\mathcal{B_p(r)} \\
&= \int_{\mathbb{R}^p} \mathbb{1}_{\lVert x \rVert \leq r} \,dx 
\end{align}$$ </div> 
<br>
<br>

$$\text{let's } g(x) = (rx_1,...,rx_p) \\
\phi_g(x) = r\mathcal{I}_p \\
|\det \phi_g(x)| = r^p$$
<br>
<div> $$\begin{align}
V_p(r) &= \int_{\mathbb{R}^p} \mathbb{1}_{|r| \lVert x \rVert \leq r}r^p \,dx \quad \text{ by substitution } \\
&=r^p \int_{\mathbb{R}^p} \mathbb{1}_{ \lVert x \rVert \leq 1} \,dx \\
&=r^pV_p(1)
\end{align} $$ </div>

<br>
<span>$$ \text{with } X^{(p)} \sim \mathcal{U}(\mathcal{B_p(1)}) \text{ we have } f_X(x) = \Large\frac{\mathbb{1}_{ \lVert x \rVert \leq 1} }{V_p(1)}$$ </span>

$$\text{and } P( \lVert X^{(p)} \rVert \leq r) = 1 \quad (\text{    if } r > 1) $$
<br>
$$\text{ if } r \leq 1 :$$

<div>$$\begin{align}
P( \lVert X^{(p)} \rVert \leq r) &= \int_{\mathbb{R}^p} \mathbb{1}_{ \lVert x \rVert \leq r} \frac{\mathbb{1}_{ \lVert x \rVert \leq 1} }{V_p(1)} \,dx \\
&=\frac{1}{V_p(1)} \int_{\mathbb{R}^p} \mathbb{1}_{ \lVert x \rVert \leq r}\,dx \\
&=r^p
\end{align} $$</div>

<span> $$\text{Then we can derive } E(\lVert X^{(p)} \rVert^2)$$ </span>

<div> $$\begin{align}
E(\lVert X^{(p)} \rVert^2) &= \int_{\mathbb{R}^+} P(\lVert X^{(p)} \rVert^2 > t) \,dt \\
&=\int_{\mathbb{R}^+} (1 - P(\lVert X^{(p)} \rVert \leq \sqrt{t})) \,dt \\
&=\int_{0}^1 1 - t^{\frac{p}{2}} \,dt \\
&= \frac{p}{p+2}
\end{align} $$ </div>

Hence :
<br>
<div> $$E(\lVert X^{(p)} \rVert^2) \underset{p \to +\infty}{\overset{}{\longrightarrow}}1 $$ </div>
