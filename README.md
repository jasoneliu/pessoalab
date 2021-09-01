# About
A documentation of my 2020 summer internship at the [Laboratory of Cognition & Emotion](https://lce.umd.edu/).
For detailed reports of my results with data visualizations, please visit [my website for this internship](https://jasoneliu.github.io/pessoalab/).

# Public Datasets
The internship began with learning to implement machine learning models with scikit-learn and PyTorch. Large public datasets such as MNIST and IMDb were used for classification.

## MNIST Digit Classification
The MNIST dataset is a collection of 70,000 28x28 pixel grayscale images of handwritten digits (0-9). The goal was to create models to accurately predict the digit contained in a given image.  
The following machine learning models were used:  
- Logistic regression
- Support vector machine (SVM)
- Naive Bayes classifier
- Ensemble methods
    - Random forest 
    - Gradient boosting
- Neural networks
    - Multilayer perceptron (MLP)
    - Convolutional neural network (CNN)

## IMDb Sentiment Analysis
The IMDb Movie Review Dataset contains 50,000 reviews, split equally into positive and negative reviews. The goal was to create models to accurately predict the sentiment (positive or negative) of a given review.  
The following deep learning models were used:  
- Vanilla recurrent neural network (RNN)
- Long short-term memory (LSTM)

# Capturing Brain Dynamics in Movie-Watching fMRI Data with Deep Learning Models
Movie-watching fMRI data were obtained from the Human Connectome Project. Participants were scanned while they watched 15 distinct movie excerpts. Principal component analysis (PCA) was used to reduce the trajectory data to three dimensions. These three dimensional trajectories were then used for classification.  
The following deep learning models were used:  
- Multilayer perceptron (MLP)
- Long short-term memory (LSTM)
- Temporal convolutional network (TCN)
- Convolutional neural network (CNN)

## Abstract
Functional Magnetic Resonance Imaging (fMRI) data, though captured temporally, is often processed statically to improve computation of highly dimensional data. In the process, static methods fail to effectively process the dynamics of a system. We proposed a dimensionality reduction method based on long short-term memory (LSTM) hidden states that was capable of creating low-dimensional trajectories that capture the most essential properties of brain dynamics. At as low as three dimensions, we found that the distinct spatiotemporal patterns in naturalistic movie-watching fMRI data can be effectively captured and classified (>90% for 15-way classification) with multiple dynamic models including LSTMs, multilayer perceptrons (MLP), and temporal convolutional networks (TCN) in order of increasing classification accuracy. Static models such as convolutional neural nets (CNN) showed a significantly lower classification accuracy than the three dynamic models, demonstrating the importance of temporal information in fMRI data. We also explored the effects of smoothing noisy trajectories with B-splines, which improved visualization of 3D trajectories but not accuracy. Further, B-spline basis function coefficients were used to reduce the 3D trajectory input to a third of the original size while only yielding a 3-4% drop in classification accuracy. While we have shown the effectiveness of various models and dimensionality reduction techniques with naturalistic stimuli in a primarily perceptual task, it is necessary to further test their capabilities in other tasks including motor, cognitive, and emotional.

## Full Paper
Access the full paper [here](https://drive.google.com/file/d/1E8KUepuuIF8RqZlzsyZt51nIyhlEty9y/view?usp=sharing). Please give it some time to load.
