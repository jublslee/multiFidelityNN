# multiFidelityNN

## Part 1: Project Description
Recently, predictive modeling has been drawing significant interest in various applications, particularly in scenarios where time-critical decisions are essential. A compelling issue in this context is to predict quantities of interest accurately when the application is intertwined with time-critical decision. Accurate prediction of specific properties require real data (a.k.a. high-fidelity data) however such physical data can be expensive, sensitive, and extremely time-consuming. A smart approach to this issue is to generate or take advantage of a giant chunk of simulated/cheap data (low-fidelity data), take a couple few of the high-fidelity data, and train the combination of both high and low-fidelity data (prevalentely known as multi-fidelity data) with a neural network. The major benefit of such multi-fidelity data aggregation using neural networks is that it significantly reduces the cost and the time of computational experiments. For this project, we propose to reproduce this idea with appropriate neural networks and examine this through multiple cases which incorporate interesting physical problems. Specifically, for this project, we will be using data generated from reaction diffusion equations (used frequently in modeling diffusion of particles or handling reactions between particles) and feed the data into our neural network system to predict the quantity of interest at any coefficient combination in a specific region accurately. 

The general process of our project proceeds as follows:

(a) Data Preparation
Prepare the data so that we have the appropriate testing, training, and validation data to feed in the implemented neural network architecture. For our application, we will (1) generate quantitative data using carefully defined mathematical equations for our network validation and (2) split our prepared data, sampled from a random 2D grid region and from (1), to testing, training, and validation data.  

(b) Design and Implement Neural Network
In this process, developing a well functioning neural network will be crucial. For our case, our network will need to capture the function/pattern between the diffusion coefficients and the quantity of interest. Accordingly, convolutional neural networks are expected to be used with additional graph or linear networks if necessary to achieve such goal.

(c) Validate Neural Network Architecture with Simple Example Case
Prior to applying the neural network to the main dataset, we examine our architecture with a simple example where low and high-fidelity are generated by carefully created mathematical functions. Once the calculated error measure between the predicted and the real data (high-fidelity model) is determined to be acceptable, we proceed to our final step.

(d) Application (Reaction Diffusion Equation)
Finally, we feed our neural network architecture our prepared data and measure the MSE of: (1) low-fidelity function, (2) high-fidelity function, (3) the real data (high-fidelity model) against the validation data set.

In summary, this project involves a systematic approach to leverage multi-fidelity data aggregation using neural networks for predicting quantities of interest; specifically, data generated from reaction diffusion equations. The application of this methodology to reaction-diffusion equations provides a practical and illustrative context for exploring the capabilities and benefits of this approach in real-world scenarios such as in understanding spatial patterns in dynamic systems utilizing a combination of mathematical framework and a fine neural network architecture.

## Part 2: Datasets

For our simple example case [1], explained as above from step (c), we approximate a one-dimensional function based on data from high and low fidelities where function is continuous and both level of fidelity generator functions are linearly correlated. 
Each level of fidelities are generated from the following functions: <br>
$y_L(x) = A(6x-2)^2sin(12x-4) + B(x-0.5) + C, x \in [0,1]$ <br>
$y_H(x) = (6x-2)^2sin(12x-4)$ <br>
where y_L(x) generates low-fidelity data, and y_H(x) generates high-fidelity data (true function). In addition, we let A = 0.5, B = 10, and C = -5. Note that the training data at the low- and high-fidelity level are respectively $x_{L}$ = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1} and $x_{H}$ = {0, 0.4, 0.6, 1}. <br>
We test our neural network's functional by first using only the high-fidelity data generated from $y_H(x)$ and comparing it with the one feeding low-fidelity data generated from $y_L(x)$ to ensure that our neural network successfully predicts the true function when multi-fidelity data are fed. Note that when only using high-fidelity data, we only will need to keeps the layers of the network where high-fidelity data are processed. 

In addition, we test if the model can capture complex nonlinear correlations between low and high fidelity generator functions. The corresponding functions are the following:<br>
$y_L(x) = sin(8\pi x), x \in [0,1]$ <br>
$y_H(x) = (x-\sqrt{2})y_L^2$ <br>
where we plan to employ uniformly distributed 51 points to generate low fidelity data and 12 data points to generate high fidelity data as the training data.

Note that if our neural network successfully works on the two simple example cases in a short amount of time, we plan to test functionality through inverse PDE problems with nonlinearities (IF we are ahead of time). 

Once we finish confirming the functional of our neural network, we proceed to 
our reaction diffusion application problem to derive and predict the solution of the reaction-diffusion PDE. In this case, we consider the 2D reaction-diffusion equation satisfying a PDE in the form as the following: <br>
$\partial_t u = D_u \partial_{xx}u + D_u \partial_{yy}u + R_u$ <br>
$\partial_t v = D_v \partial_{xx}v + D_v \partial_{yy}v + R_v$ <br>
where $u = u(t,x,y)$ is the activator and $v = v(t,x,y)$ is the inhibitor. Note that activator and inhibitor is simply two types of chemical components involved in our system which is generally used to explain pattern formation dynamics or various spatial structures. To be more detailed, activator promotes production of others and itself so it is likely to have positive influence on reaction terms where on the other hand, invibitors suppresses production of itself and others so is likely to have negative influence on the reaction terms. So, the interaction between the activator and inhibitor stimulates unique spatial patterns and structures. Accordingly, $R_u = R_u(u,v)$ and $R_v = R_v(u,v)$ are respectively the activator and inhibitor reaction function. <br>
Furthermore, for the boundary conditions, we consider a no-flow Neumann boundary condition (i.e. $D_u \partial_{x}u = 0, D_v \partial_{x}v = 0, D_u \partial_{y}u = 0, D_v \partial_{y}v = 0$) where Neumann boundary condition is simply a certain type of boundary condition used in PDEs and by no-flow, this generally infers that there is no net flow across the boundary (i.e. zero net movement of the substance across the boundary). <br>
Also, we consider uncertain initial condition $u(0,x,y),v(0,x,y) ~ N(0,1) \forall x,y$ where $D_u$ and $D_v$ are the corresponding diffusion coefficients. <br>
For our reaction functions, Fitzhugh-Nagoumo equations are used: <br>
$R_u(u,v) = u - u^3 - k - v$ <br>
$R_v(u,v) = u - v$ <br>
where $k = 5*10^{-3}$. <br>
This particular set of equations exhibit interaction of variables with different excitement levels and can express and display the transition between resting and excited states under certain conditions.

Specifically, we are interest in predicting a scalar value of the mean concentration as a function of the diffusion coefficients $D_{u}$ and $D_{v}$ where the scalar value represents the mean concentration of our inhibitor u in the region [0,1]x[-1,0] at t = 1. 

With respect to the problem set, we generate two predictors using functionalities from the PDEBench library[2], where the modified code is attached to this repository, which are each the high-fidelity predictor using a discretized domain of size 128x128 and the low-fidelity predictor using a 64x64 spatial mesh.

Note that there might be slight modification of the specific domain size due to computational cost throughout the project application.

[Source]

[1] arXiv:1903.00104 [physics.comp-ph]

[2] https://github.com/pdebench/PDEBench