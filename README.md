## Designing recurrent neural networks for explainability 
![](https://travis-ci.org/heytitle/thesis-designing-recurrent-neural-networks-for-explainability.svg?branch=master)

Standard (non-LSTM) recurrent neural networks have been challenging to train, but special optimization techniques such as heavy momentum makes this possible. However, the potentially strong entangling of features that results from this difficult optimization problem can cause deep Taylor or LRP-type to perform rather poorly due to their lack of global scope. LSTM networks are an alternative, but their gating function make them hard to explain by deep Taylor LRP in a fully principled manner. Ideally, the RNN should be expressible as a deep ReLU network, but also be reasonably disentangled to let deep Taylor LRP perform reasonably. The goal of this thesis will be to enrich the structure of the RNN with more layers to better isolate the recurrent mechanism from the representational part of the model. Various RNN structures will be tested with the quality of explanations as a selection criterion. Technically, the thesis will consist of the following steps:

- Implement Deep Taylor LRP for general RNNs.
- Train various RNN architectures on several datasets of interest.
- Compare the quality of explanations produced for the these RNN architectures.


Supervisor: **[Dr. Grégoire Montavon](http://gregoire.montavon.name)**

