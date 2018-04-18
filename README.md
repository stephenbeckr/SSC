
# Sparse Subspace Clustering (SSC) algorithms

This repository contains Matlab code to implement ADMM and proximal gradient algorithms to solve the SSC clustering problem (several variants, including variants for affine subspaces). The SSC model we work with is based on the well-known work of [Elhamifar and Vidal's Sparse "Subspace Clustering: Algorithm, Theory, and Applications" (IEEE Trans. on PAMI)](https://scholar.google.com/scholar?cluster=7262850065108933522&hl=en&as_sdt=0,6&as_vis=1) (and their code is at their [VisionLab website ](http://vision.jhu.edu/code/) 
as well as their [JHUVisionLab Github site](https://github.com/JHUVisionLab/SSC-using-ADMM) ).

The paper that explains this code is [Efficient Solvers for Sparse Subspace Clustering](http://arxiv.org/abs/1804.06291) (Pourkamali-Anaraki and Becker, 2018)

## Why use this code?

The new ADMM code is much faster, as it scales like *O(n^2)* instead of *O(n^3)*

![Scaling](figs/Fig4.jpg?raw=true "Good Scaling")

The proximal gradient code doesn't have the extra `rho` parameter that ADMM algorithms need. The ADMM algorithms are quite sensitive to this parameter, as the following experiment shows:

![Parameters](figs/Fig2.jpg?raw=true "Parameter Issues")

## Dependencies
To run the l1 proximal gradient descent code, you need a recent copy of the [TFOCS package](https://github.com/cvxr/TFOCS/).  
One of the scripts uses [CVX](https://github.com/cvxr/CVX) as well, but CVX is not needed to run any of the SSC functions, only for comparison

## Authors
The authors are [Stephen Becker](http://amath.colorado.edu/faculty/becker/) and [Farhad Pourkamali-Anaraki](http://www.pourkamali.com/) (University of Colorado Applied Math)

This README from April 2018. Thanks to https://stackedit.io/app for editing markup
