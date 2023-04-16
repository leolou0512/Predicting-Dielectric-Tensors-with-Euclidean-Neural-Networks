# Introduction, Theory and Model Architecture of the Euclidean Convolutional Neural Network

Author: Yuchen Lou
# Table of Contents:
* [Introduction](#Introduction)
  * [Requirements](#requirements)
  * [Setup](#setup)
* [Background](#Background)
  * [Graph Neural Networks](#GNN)
  * [Equivariant Machine Learning](#EML)
* [Theory](#Theory)
  * [Graph Neural Networks](#GNN)
  * [Equivariance and Invariance](#EandI)
  * [Irreducible Representations and Tensor Products](#irrep)
  * [Spherical Harmonics](#sh)

## Introduction <a class="anchor" id="Introduction"></a>
This project aims to construct an equivariant graph convolutional neural network to predict the dielectric tensors for periodic materials from crystal structures. An equivariant network achieves higher performance with a smaller dataset compared to traditional deep neural networks, and the predictions are guaranteed to be either invariant or equivariant. Notably, this model is the first of its kind to predict dielectric tensors instead of just dielectric constants. The dataset used in this project is taken from the open-access database Materials Project (https://materialsproject.org/). The code used in this project is based on 'Direct Prediction of Phonon DoS with Euclidean Neural Networks' (https://github.com/ninarina12/phononDoS_tutorial).

### Requirements <a class="anchor" id="requirements"></a>
PyTorch 1.13.1 with cuda  
torch_geometric    
torch_scatter  
torch_cluster  
pandas  
numpy  
ase (Atomic Simulation Environment)  
e3nn  

### Setup <a class="anchor" id="setup"></a>
1. Open anaconda command prompt and create a virtual environment for this project:   
```
conda create -n e3nn python=3.9

conda activate e3nn
```
  
2. Install the relevant packages:     
```
!pip install ase e3nn

!pip install torch-scatter torch-cluster torch-sparse torch-spline-conv -f https://pytorch-geometric.com/whl/torch-$(python -c "import torch; print(torch.__version__)").html 

!pip install torch-geometric   

```

3. Launch Jupyter lab and navigate to main.ipynb:
```
jupyter lab
```

4. Follow instructions in main.ipynb.

### Reference
Chen, Zhantao, Nina Andrejevic, Tess Smidt, Zhiwei Ding, Qian Xu, Yen-Ting Chi, Quynh T. Nguyen, Ahmet Alatas, Jing Kong, and Mingda Li. “Direct Prediction of Phonon Density of States With Euclidean Neural Networks.” Advanced Science 8, no. 12 (2021): 2004214. https://doi.org/10.1002/advs.202004214.

Xie, Tian, and Jeffrey C. Grossman. “Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties.” Physical Review Letters 120, no. 14 (April 6, 2018): 145301. https://doi.org/10.1103/PhysRevLett.120.145301.

Thomas, Nathaniel, Tess Smidt, Steven Kearnes, Lusann Yang, Li Li, Kai Kohlhoff, and Patrick Riley. “Tensor Field Networks: Rotation- and Translation-Equivariant Neural Networks for 3D Point Clouds.” arXiv, May 18, 2018. https://doi.org/10.48550/arXiv.1802.08219.

Liao, Yi-Lun, and Tess Smidt. “Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs.” arXiv, June 23, 2022. https://doi.org/10.48550/arXiv.2206.11990.



# ============================
# See below for detailed background and theory

## Background <a class="anchor" id="Background"></a>
In recent years, efforts were made to catalogue the dielectric constants for whole databases, so that the materials can be pre-screened, constraining the experiments only to the most promising candidates. Obtaining dielectric constants (or tensors) are mostly done using three ways: Experiments, first-principle calculations, and machine learning. Due to the increasing availability of large datasets and breakthroughs in model architectures, machine learning has become a useful tool in material science. With the help of machine learning, researchers can train a model with a dataset, then use the model to make property prediction for new materials at virtually no computational cost. This is very appealing to the material scientists, since this is a reasonable alternative to run time consuming and expensive calculations.

### Graph Neural Networks <a class="anchor" id="GNN"></a>
Graph is a data structure that expresses both a set off entities and the relationship between entities. The entities are represented as nodes and edges are formed between nodes to present some sort of relationship. Neural networks that operate on graph domains are called graph neural networks (GNN). GNNs can be thought of as convolutional neural networks (CNN) on irregular shaped graphs. Recently, GNNs have attracted some attention particularly in the field of material science and chemistry. One of the main reasons for this is due to GNN's flexibility. Traditional CNNs can only operate on regular data structures such as images, which re 2-dimensional grid, but GNNs can operate on point clouds and other complex data structures in the non-Euclidean domain. In material science and chemistry, graphs can encode the intrinsic geometric information into the network. This is done by using the full molecular structure and geometry as input, representing atoms as nodes with a set of attributes (such as atom type, number of covalent bonds, formal charge etc.), and bonds as edges with information such as bond-type and conjugation. Such representation of features are informative to the machine learning algorithm, and are in good agreement with chemical intuition. GNNs have shown potential to complement or even replace the traditional composition-based feature engineering altogether. GNNs have the added benefit of being a natural and intuitive way of representing data, making GNNs easily comprehensible to humans.

### Equivariant Machine Learning <a class="anchor" id="EML"></a>
One common problem with neural networks is that for models with large number of parameters, a large quantity of data is required to generate an observational bias, which means for the model to 'understand' the underlying laws of physics. This method is called data augmentation. In scientific contexts, data is usually obtained through experiments or expensive simulations, and the amount of data produced is sometimes insufficient to train a conventional neural network effectively. To mitigate this challenge, Thomas and Smidt introduced equivariant models. Equivariance is a property of an operator, which means the output of the operator will transform in exactly the same way the input is transformed.
An equivariant neural network is significantly more data efficient compared to non-equivariant counterparts. The model introduced by Thomas and Smidt haven been shown to make meaningful predictions even when the dataset is as small as 1,000 molecules. The reason for the data efficiency is that for non-equivariant models, the same features in extracted from the input that are in different orientations (such as rotation, translation, or inversion) are learned separately. This resulted in redundancy in the model, since the model is repeating the learning of the same feature but at different orientations. But an equivariant model always learns the same type of feature together, regardless of the orientation of the input data. Equivariant model also guarantees output equivariance. For example, when predicting a scalar, the output generated by an input structure will be the same as the output generated by the same but transformed input structure. Non-equivariant models suffer from this inconsistency when given original vs transformed input structures. For example, it is undesirable for the model to predict a slightly different band gap, when the input structure is only rotated for 90$^{\circ}$. CNNs achieved great success in computer vision largely due to its translational-equivariance. This means that a feature at the top of the image will produce the same output as the same feature at anywhere else on the image.

## Theory <a class="anchor" id="Theory"></a>
### Group Theory <a class="anchor" id="Group Theory"></a>
A group consists of a set (G) and a binary operation that has an inverse and identity, and is both associative and closed. The group concerned in this project is the Euclidean group in three dimensions: E(3), which includes 3D rotation, inversion and translation. To perform these group actions on a certain vector space X, the group actions need to be expressed as representations, which are matrices specific to vector space X. A representation (D(g)) then can be implemented simply using matrix multiplication.

### Equivariance and Invariance <a class="anchor" id="EandI"></a>
Equivariance is a property of an operator that preserves the symmetry of the object under a group of transformations. A function f that maps one vector space to another (f: X → Y) is equivariant with respect to representation D^X(g) and D^Y(g) if for all group elements in a group (g ∈ G) and input (x ∈ X),

$$
\begin{equation}
    f(D^\mathcal{X}(g)x) = D^\mathcal{Y}(g)f(x)
\end{equation}
$$

Invariance is a special type of equivariance, where the output remains the same before or after $f$. In atomic systems, physical properties like dielectric tensors remains the same regardless of our choice of unit cell and rotations. Comparing a material to itself after a rotation, one would expect scalar properties like energy or band gap of a material to be invariant (to remain the same), and tensorial properties such as force or dielectric tensors to be equivariant (to rotate accordingly). Equivariance is very a powerful property in materials machine learning, as the output will always be consistent, no matter the input unit cell orientation. Equivariant models don't need to learn redundantly from multiple orientations, but rather identify the general trends irrespective of the input orientation. As a result, equivariant models often demonstrates a superior ability to achieve higher accuracies given the same dataset.

Due to the associativity of representations, 

$$
\begin{equation}
    D(g)D(h)=D(gh)
\end{equation} 
$$

For all $g,h \in G$, composing multiple equivariant transformations $g$ and $h$ will result in an equivariant transformation. The deep neural network trained in this project only consists of equivariant layers, thus the network will overall be equivariant. 

### Irreducible Representations and Tensor Products <a class="anchor" id="irrep"></a>
A representation in a vector space is irreducible if it cannot be decomposed into a concatenation of smaller and independent representations. These representations are called **ir**reducible **rep**resentations (irreps).

The group of rotation in 3D space is called SO(3). In SO(3), for a (2L+1)-dimensional space, there exists an irreps matrices of dimension (2L+1)-by-(2L+1), called Wigner-D matrices. The order of an irrep L can be thought of as the angular momentum, and takes a non-negative integer value. For example, in 3D space (L=1), a 3x3 representation generated from the tensor products of two vectors:

$$
\begin{pmatrix}
x_1 \\
y_1 \\
z_1 
\end{pmatrix}
\otimes
\begin{pmatrix}
x_2 \\
y_2 \\
z_2 
\end{pmatrix}

can be decomposed into $L=0$ part, $L=1$ part and $L=2$ part through a process called Wigner decomposition. The $L=0$ part is the trace of the 3x3 matrix with dimension of 1, this is called a scalar. The physical intuition is that when the two vector rotates, the scalar product of the two vectors will not change. The $L=1$ part is the cross product of (x1, y1, z1) and (x2, y2, z2) with dimension of 3, this is called a vector. If the two vectors are rotated, the cross product will also rotate. The $L=2$ part is the symmetric traceless part of the matrix and have a dimension of 5. This part does not have a specific name. All three representations generated from the original representation cannot be reduce anymore. The elements of these irreps are index with m, where -L <= m <= L. See below for details:

$$
\begin{pmatrix}
x_1 x_2 & x_1 y_2 & x_1 z_2 \\
y_1 x_2 & y_1 y_2 & y_1 z_2 \\
z_1 x_2 & z_1 y_2 & z_1 z_2
\end{pmatrix} \longrightarrow (x_1 x_2 + y_1 y_2 + z_1 z_2) \oplus
\begin{pmatrix}
y_1 z_2 - z_1 y_2 \\
z_1 x_2 - x_1 z_2 \\
x_1 y_2 - y_1 x_2
\end{pmatrix} \oplus
\begin{pmatrix}
c(x_1 z_2 + z_1 x_2) \\
c(x_1 y_2 + y_1 x_2) \\
2y_1 y_2 - x_1 x_2 - z_1 z_2 \\
c(y_1 z_2 + z_1 y_2) \\
c(z_1 z_2 - x_1 x_2)
\end{pmatrix}
$$

The group formed by combining the inversion group $\mathbb{Z}_2$ with $SO(3)$ is called $O(3)$. There are twice as many irreps in $O(3)$ since every irrep in $SO(3)$ now has an even form and an odd form. The even irreps do not change sign under parity while the odd irreps do. These irreps are denoted by $l$ followed by parity, for instance, 1e for even, 1o for odd. The dielectric tensors are symmetric 3x3 matrices, and after decomposition to irreps, we get $1 \times 0e+1 \times 2e$. $L=1$ is missing since the $L=1$ component (cross-product) is always zero for symmetric matrices.

In the model, irreps are combined using tensor products. To calculate the $m_3$ element of a type-$L_3$ output of a type-$L_1$ vector $f^{(L_{1})}$ with a type-$L_2$ vector $g^{(L_{2})}$ using tensor product, we use this equation:

$$
\begin{equation}
    h^{(L_{3})}_{m_{3}} = (f^{(L_{1})} \otimes g^{(L_{2})})_{m_{3}} = \sum_{m_1=-L_1}^{L_1} \sum_{m_2=-L_2}^{L_2} C^{(L_{3},m_{3})}_{(L_{1},m_{1})(L_{2},m_{2})} f^{(L_{1})}_{m_{1}} g^{(L_{2})}_{m_{2}}
\end{equation}
$$

The order $L$ of the irreps after Clebsch-Gordan decomposition can be summarized with the following equation: 

$$
\begin{equation}
    L_1 \otimes L_2=|L_1 - L_2|\oplus \dots \oplus (L_1 + L_2)
\end{equation}
$$

This is due to the fact that Clebsch-Gordan coefficients are only non-zero between this boundary. The previous example of combining two 3D vectors can be thought of as $1 \oplus 1=0 \oplus 1 \oplus 2$.
The network in this project takes irreps as input, and outputs irreps. Input irreps interacts with other irreps in the network using tensor products as it propagates through the network. There are various types of tensor products in the e3nn.o3 package, such as o3.TensorProducts, o3.FullTensorProducts and o3.FullyConnectedTensorProducts. o3.FullTensorProducts creates and returns all possible combinations of the two pairs of input irreps, resulting in output irreps that are independent from each other, i.e. the outputs not mixed. The multiplicity of the outputs are the product of the multiplicity of the two parent irreps (e.g. $3\times0e \otimes 5\times0e = 15\times0e$). o3.FullyConnectedTensorProducts can be seen as o3.FullTensorProducts followed by a fully connected layer. In addition to specifying two input irreps, output irreps are required as an argument. Then for each output irrep specified, the function will create a weighted sum of all compatible irreps, the weights of which are learnable. This allows the output to have any multiplicity. \boxed{\small \text{o3.TensorProducts}} creates tensor products with parameterized paths. This function has the most flexibility, allowing users to specify multiplicity of the outputs and toggle learnable weights independently.  

### Spherical Harmonics <a class="anchor" id="sh"></a>

Spherical harmonics are a set of basis functions that originates from angular part of the solution of the Laplace equation in spherical coordinates. Square-integrable functions are defined as functions that satisfies: 

$$
\begin{equation}
    \int^{2\pi}_{0} \int^{\pi}_{0} |f(\theta, \phi)|^{2} sin\theta d\theta d\phi < \infty
\end{equation}
$$

where $\theta$ $\phi$ are polar and azimuthal angles. Any square-integrable functions on the unit sphere can be expressed as a linear combination of spherical harmonics. Euclidean vectors $\vec{r}$ in $\mathbb{R}^{3}$ can be projected into irreps using spherical harmonics. For example, the spherical harmonics for $l=1$ are: 

$$
\begin{align}
    Y_{1,-1}(\theta,\phi) &= \sqrt{\frac{3}{8\pi}}\sin\theta e^{-i\phi}, \\
    Y_{1,0}(\theta,\phi) &= \sqrt{\frac{3}{4\pi}}\cos\theta, \\
    Y_{1,1}(\theta,\phi) &= -\sqrt{\frac{3}{8\pi}}\sin\theta e^{i\phi}
\end{align}
$$

and the weight $c_{l,m}$ for each $Y_{l,m}$ can be worked out via:

$$
\begin{align}
    c_{1,-1} &= \sqrt{\frac{4\pi}{3}}\int Y_{1,-1}(\theta,\phi)^* (x,y,z) \sin\theta d\theta d\phi, \\
    c_{1,0} &= \sqrt{\frac{4\pi}{3}}\int Y_{1,0}(\theta,\phi)^* (x,y,z) \sin\theta d\theta d\phi, \\
    c_{1,1} &= \sqrt{\frac{4\pi}{3}}\int Y_{1,1}(\theta,\phi)^* (x,y,z) \sin\theta d\theta d\phi 
\end{align} 
$$

where the * represents complex conjugate.

Spherical harmonics possess rotational equivariance, which makes them a good choice to make up the angular part of the convolutional kernel. The convolutional kernel used in this project consists of a learned radial part and spherical harmonics angular part. Detailed explanation of convolution kernel can be found in Model Architecture section.
