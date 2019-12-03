# Channel-Equalization

# Problem Description
When a sequence of bits is transmitted through a channel, they may get distorted because of intersymbol interference (ISI) & noise. The task of channel equalization is to regenerate originally transmitted bit sequence from the received, possibly distorted values.
Assume, the k-th transmitted bit is Ik & received sample is xk. If we assume ISI spans over n successive bits, xk can be expressed as follows.
xk = f(Ik,Ik−1,Ik−2,...,Ik−(n+1)) + ηk	(1)

Here f(.) represents the action of the channel & denotes ηk noise for k-th bit. We assume f(.) to be a linear function represented as:
n−1
f(Ik,Ik−1,Ik−2,...,Ik−(n+1)) = XwjIk−j	(2)
j=0

We can also assume noise to follow a normal distribution with mean = 0 & variance = σ2.
The task of equalizer is to predict k-th transmitted bit from previous l successively received samples.
Iˆk = g(xk,xk−1,xk−2,...,xk−(l+1))	(3)

In the above equation, g(.) is the equalizer method. There are several ways for implementing it such as using markov chain model or cluster based approach. In this assignment you have to implement the markov chain model with proper selection of states & Viterbi algorithm for predicting transmitted bit sequence.

# Input
Read the parameters from a file named ”config.txt”. The description of this file is as follows.
•	The first line contains two positive integers n and l.
•	Next line contains n space separated real numbers denoting w0, w1, ..., wn−1
•       Next line contains a single real number denoting variance of noise, σ2

From another file named ”train.txt”, read a single bit string consisting of 0’s and 1’s, lets call it trainBits.
Finally, read another bit string from ”test.txt” file, lets call it testBits.

# Sample
config.txt
3 2
0.7 0.5 0.1 0.225

train.txt
00000101001110010111011110100111001011010011100101110111

test.txt
0110100011

Here, n = 3, l = 2, w0 = 0.7, w1 = 0.5, w2 = 0.1, σ2 = 0.225

trainBits = 00000101001110010111011110100111001011010011100101110111 testBits = 0110100011

# Tasks

1. Construct a markov chain model for which the following is required:
•	Define all possible states based on value of n.

•	Calculate prior probabilities of all states from trainBits. Note that each consecutive n bits in trainBits refer to a state.             [ Hint : Prior probability of a state is related to the number of appearances of the state in trainBits]

•	Calcuate transition probabilities from trainBits. 
        [Hint: Transistion probability from state A to state B is related to the count of state transition from A to B in trainBits.]
        
•	Assume observation probability follows normal distribution & calculate the means of observations.


2.	Transmit testBits, calculate xk’s and use Viterbi algorithm on your markov model to reconstruct originally transmitted bits.


3.	Calculate accuracy by comparing originally transmitted and predicted bit sequence.



