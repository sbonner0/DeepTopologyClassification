# Deep Topology Classification

This repo contains code and datasets for the IEEE Big Data 2016 paper entitled - " Deep Topology Classification: A New Approach for Massive Graph Classification". It provides a feed forward neural network model for the classification of complex graphs based upon their feature vectors.

## 2020 Update

The original model has been reimplemented in pytorch and has been upgraded to support Python 3. Please note that the original Keras/TF implementations can be found in the `legacy` folder, however these have not been updated for Python 3 support. These files are being preserved so the original experiments from the paper can be replicated. 

Going forward, only the GFP feature extraction code, pytorch based model and training will be maintained. 

## Requirements

This code has been tested on Python 3.8.5+ and requires the following packages to function correctly:

- torch 1.4+
- graph-tool 2.33+
- networkx 1.5+

Implements are also provided using the Skorch toolkit for pytorch and scikit-learn integration, in which case there is additional requirements:

- skorch 0.8+

## Datasets

There are two datasets included in the python pickle format. One contains graphs from multiple graph generation methods and as such has multi classes contained within. The second dataset contains just two classes and as such is a binary dataset. 

Users own datasets can easily be used by generating fingerprint feature vectors from labelled graphs which you would like to be classified. A DTC model can then be trained.

## Cite

Please cite the associated papers for this work if you use this code:

```
@inproceedings{bonner2016deep,
  title={Deep topology classification: A new approach for massive graph classification},
  author={Bonner, Stephen and Brennan, John and Theodoropoulos, Georgios and Kureshi, Ibad and McGough, Andrew Stephen},
  booktitle={2016 IEEE International Conference on Big Data (Big Data)},
  pages={3290--3297},
  year={2016},
  organization={IEEE}
}
```