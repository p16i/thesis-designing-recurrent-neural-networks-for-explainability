## Designing Recurrent Neural Networks for Explainability
[![To Travis](https://travis-ci.org/heytitle/thesis-designing-recurrent-neural-networks-for-explainability.svg?branch=master)](https://travis-ci.org/heytitle/thesis-designing-recurrent-neural-networks-for-explainability)

Supervised by : **[Prof. Dr. Klaus-Robert Müller](http://www.ml.tu-berlin.de/menue/members/klaus_robert_mueller/)**, **[Dr. Grégoire Montavon](http://gregoire.montavon.name)**

### Abstract
Neural networks (NNs) are becoming increasingly popular and are being used in many applications nowadays. Despite achieving state-of-the-art performance in various domains, NNs are still considered as black boxes, since it is difficult to explain how these models map input to output and achieve high accuracy.  Consequently, several techniques have been proposed to explain the  predictions from NNs. Such methods include sensitivity analysis (SA), guided backprop (GB), Layer-wise Relevance Propagation (LRP), and deep Taylor decomposition (DTD).  Recurrent neural networks (RNNs) are NNs that have a recurrent mechanism, which makes them well suited for modeling sequential data.  

Unlike feedforward architectures, RNNs need to learn and summarize data representations in order to utilize them across time steps. Therefore, well-structured RNNs that can isolate the recurrent mechanism from the representational part of the model will have an advantage of being more explainable. In this thesis, we apply the explanation techniques mentioned above to RNNs. We extensively study the impact of different RNN architectures and configurations on the quality of explanations. Our experiments are based on artificial classification problems constructed from MNIST and FashionMNIST datasets. We use cosine similarity to quantify the evaluation results. Our results indicate that the quality of explanations from different RNNs, achieving comparable accuracies, can be notably different.

Based on our evaluations, the deeper and LSTM-type RNNs have more explainable predictions regardless of the explanation methods.  Convolutional and pooling layers, and the stationary dropout techniques are other factors influencing the quality of explanations. We also find that some explanation techniques are more sensitive to the RNN architecture. We propose a modification to the LSTM architecture enabling the model to be explained by the mentioned techniques. The modified architecture shows significantly improved explainability without adversely affecting predictive performance.

[[ **PDF** ]](http://bit.ly/2KOe8ZF)

### Poster
**Presented at Data Science Summer School (DS^3) 2018,  École Polytechnique, Paris**
![](https://i.imgur.com/FduaFZP.jpg)


### Other repos
- Writing https://github.com/heytitle/thesis-writing
- Presentation https://github.com/heytitle/thesis-presentation
