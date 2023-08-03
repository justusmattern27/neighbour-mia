# Neighbourhood MIAs

This is the code for the paper [Membership Inference Attacks against Language Models
via Neighbourhood Comparison](https://aclanthology.org/2023.findings-acl.719.pdf).

### Prerequisites:

To run our code, you need to have a model you want to attack (in `path_to_attack_model`)as well as a dataset consisting of training members and non members. in `attack.py`, examples for news, twitter and wikipedia data are provided. In the code, we assume that the first n lines of the text file are members and the n remaining ones are non-training-members.


### How it works:

The code will use a BERT based model to generate neighbours and compute the likelihoods of neighbours and the original texts under the probability distribution of the provided, gpt2-based attack model.
It will return these scores in a pickle file.

To parallelize the workload, you should provide a --proc-id as an argument
