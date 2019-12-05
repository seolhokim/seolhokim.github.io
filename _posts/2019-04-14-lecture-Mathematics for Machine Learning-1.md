---
layout: post
title:  "Mathematics for Machine Learning-1"
subtitle:   "기호 정리"
categories: deeplearning
tags: lecture
---

논문을 읽다 보면 항상 대충 이거지 하고 넘어갔던 부분을 상세하게 정리하기 위해서 적어놓습니다.

간단한 Symbol 정리부터

$$ a,b,c,\alpha, \beta, \gamma  : scalars $$ 

$$ \textbf{x,y,z} : vectors $$

$$ \textbf{A, B, C }: matrixs $$

$$ x^T, \textbf {X}^T : transpose $$

$$ \textbf{X}^{-1} : inverse $$

$$ <x,y> : inner\ product $$

$$ x^Ty : dot\ product $$

$$ B = (b1,b2,b3) :ordered \ tuple $$

$$ \textbf{B} = [b1,b2,b3] : matrix\ of\ column\ vectors\ stacked\ horizontally $$ vectors 쌓은 matrix

$$ \mathcal{B} = \{b1,b2,b3\} : set\ of\ (unordered)\ vectors $$

$$ \mathbb{N} : natural \ numbers $$

$$ \mathbb{Z} : integer $$

$$ \mathbb{R} : real \ number $$

$$ \mathbb{C} : complex \ number $$

$$ \mathbb{R} ^n : n-dimentional \ vector \ space \ of \ real \ numbers $$

---------------------------------------

$$ \forall x\ :\ For\ all \ x $$

$$ \exists x\ :\ There\ exists\ x $$

$$ a := b\ :\ a\ is\ defined\ as\ b$$

$$ a \propto b\ :\ a\ is\ propotional\ to\ b $$

$$ g \circ f\ :\ functional\ composition\ :\ "g\ after\ f" $$

$$ \Leftrightarrow \ :\ iff $$

$$ \mathcal{A,C}\ :\ set$$

$$ \emptyset\ :\ empty\ set $$

$$ a \in A \ :\ a\ is\ an\ element\ of\ the\ set\ \mathcal{A} $$


---------------------------------------

$$ \textbf{I}_m \ :\ identity\ matrix\ of\ size\ m\ \times m $$

$$ \textbf{0}_{m,n} \ : \ matrix\ of\ zeros\ of\ size\ m \times n $$

$$ \textbf{1}_{m,n} \ : \ matrix\ of\ ones\ of\ size\ m \times n $$

$$ dim\ : \ dimensionality\ of\ vector\ space $$

$$ rk(\textbf{A}) \ :\ rank\ of\ matrix \ \textbf{A} $$

$$ \textbf{Im}(\Phi)\ :\ image\ of\ linear\ mapping \Phi $$

$$ span \left[ \textbf{b}_{1} \right]  : span\ of\ \textbf{b}_{1} $$

$$ tr(\textbf{A})\ : \ trace\ of\ \textbf{A} $$

$$ det(\textbf{A}) \ : \ determinant\ of\ \textbf{A} $$

$$ \leftmid · \rightmid \  :  abs $$

$$ \leftparallel · \rightparallel \ :  norm $$

$$ \lambda \ : \ eigenvalue\ or\ Lagrange\ multiplier $$

$$ \textbf{E} _ {\lambda}  : \ eigenspace\ corresponding\ to \ eigenvalue\ \lambda $$


---------------------------------------

$$ \theta \ :\ parameter \ vector $$

$$ \frac{\partial f}{\partial x} : \ partial \ derivative\ of\ f\ with\ respect\ to\ x $$

$$ \bigtriangledown : \ gradient $$

$$ \mathcal{L} : \ negative\ log-likelihood $$

$$ \mathfrak{L} : \ Lagrangian $$

$$ \mathbb{Z}_{\textbf{X}} \left[ x \right] \ : \ variance\ of\ x\ with\ respect\ to\ the\ random\ variable\ X$$

$$ \mathbb{E}_{\textbf{X}} \left[ x \right] \ : \ expectation\ of\ x\ with\ respect\ to\ the\ random \ variable\ X $$

$$ \textbf{X} \sim p \ : \ random\ varaible\ \textbf{X} \ is\ distributed\ accoding\ to\ p $$

$$ \mathcal{N}(\mu , \sum) \ : \ Gaussian\ distribution\ with \ mean\ \mu , and covariance \ \sum $$








---------------------------------------
i.e. : this means

e.g. : for example

MAP : maximum a posteriori

MLE : maximum likelihood estimation


[https://mml-book.github.io/book/mml-book.pdf] 이곳의 책의 내용을 따서 정리하였습니다.
