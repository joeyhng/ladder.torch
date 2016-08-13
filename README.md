# Ladder Network
This repository contains a partial implementation of Ladder Network described in [[1]](http://arxiv.org/abs/1507.02672). 

This code is just for verifying my understanding of Ladder network.
I did not attempt to optimize the results and test extensively on different combinator functiosn.
Therefore, the results may be slightly worse than the original paper.


## Dependencies
Depends on ```dpnn``` and ```nninit```. To install, run
```
luarocks install dpnn
luarocks install nninit
```

To download the MNIST dataset:
```
wget https://s3.amazonaws.com/torch7/data/mnist.t7.tgz
tar xzf mnist.t7.tgz
rm mnist.t7.tgz
```

## Run
Only the MNIST model is implemented. An example run is as follow:
```
th main.lua -num_labels 100 -comb_func vanilla-randinit -learning_rate 0.0002
```
This gives about 98.8% accuracy, which gives similar performance as ```RandInit``` in [2].

I am not able to obtain better performance with ```Gaussian``` or ```Vanilla``` as stated in the paper.
This may be obtained by better choice of hyperparameters.


## Reference
[1] A. Rasmus, H. Valpola, M. Honkala, M. Berglund, T. Raiko. Semi-Supervised Learning with Ladder Networks. NIPS 2015.
[2] M. Pezeshki, L. Fan, P. Brakel, A. Courville, Y. Bengio. Deconstructing the Ladder Network Architecture. ICML 2016.
