# Reducing Communication Overhead in Distributed Learning via CEO

We frame the distributed deep learning as a central estimation officer (CEO) problem; starting from an initial guess for the model shared by all workers and the server, at each iteration of training, the workers refine the model parameters by training over their available dataset and send them to the server. The server merges the received information from the workers to get a better estimate of the optimum parameter of the model and sends it back to the workers. We propose using CEO framework to tackle the communication issue in distributed training. The proposed method consists of three major blocks: 
1) Dithered and nested quantization at the workers, 
2) Distributed source coding to incorporate the correlation among workers for further reduction in communication bit rate, 
3) Decoding the data received from the workers and estimating the optimum parameters at the server.

###### citing the paper
For a reference to the *Distributed Training via CEO*, please cite the following paper
* A. Abdi, and F. Fekri, "Reducing Communication Overhead via CEO in Distributed Training," *IEEE International Workshop on Signal Processing Advances in Wireless Communications (SPAWC)*, 2019, pages 1-5
