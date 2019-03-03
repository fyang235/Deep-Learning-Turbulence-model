# Deep-Learning-Turbulence-model
Use deep learning to learn a turbulence model from high fedelity data. The model can reasonably predict other turbulent flows.  

We combine [OpenFOAM](https://www.openfoam.com/) C++ code with deep neural network python code to build a deep learning turbulence model.  

### Procedures:  
1. Extract high fidelity turbulence flow information in OpenFOAM using c++ code.  
2. Preprocessing the dataset and training a mode using python code.   
3. Emmbeding the DNN weights to OpenFOAM using c++ code.  
4. Combine the deep learning turbulence model with OpenFOAM turbulent flow calculation.  

### The model makes reasonable predicitons and outperforms conventional turbulence model for coures-mash condition. Below is some features map obtained using our deep learning turbulence model:
![](https://github.com/fyang235/Deep-Learning-Turbulence-model/blob/master/gif/F1.gif)  

![](https://github.com/fyang235/Deep-Learning-Turbulence-model/blob/master/gif/F3.gif)    

![](https://github.com/fyang235/Deep-Learning-Turbulence-model/blob/master/gif/I1.gif)    

![](https://github.com/fyang235/Deep-Learning-Turbulence-model/blob/master/gif/I3.gif)    


### Checkout the [turML Model Poster.pdf](https://github.com/fyang235/Deep-Learning-Turbulence-model/blob/master/turML%20Model%20Poster.pdf) file for details.  

Enjoy!
