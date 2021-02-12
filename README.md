# demo_PMACE

# PMACE algorithm 
[//]: # (Qiuchen Zhai, Dr. Wohlberg, Prof. Charles A. Bouman, Prof. T. Gregery Buzzard<br/>)
[//]: # (Email: qzhai@purdue.edu<br/>)
[//]: # (Institution: School of Electrical and Computer Engineering, Purdue University<br/>)

## Overview
This python package implements PMACE algorithm for solving ptychographic 
reconstruction problem. 

## Requirements
The required packages are included in requirements.txt. Please build your virtualenv first, activate it and then run the following command in your shell:

```console
pip install -r requirements.txt
```


[//]: # (## Quick start)
[//]: # (Institution: School of Electrical and Computer Engineering, Purdue University<br/>)

## Files
#### 1. demos
* demo_IC_simulation.py <br/>
This demo demonstrates the generation of diffraction patterns using the digital phantom and reconstruction of complex image performed by PMACE algorithms and Wirting Flow algorithms. The IC image is included in the subfloder named 'data'.
* demo.py <br/>
This demo performs PMACE reconstruction of given dataset.
#### 2. utils
It contains code for PMACE algorithm for differnet use. 

## Other
Constructing this repository is a work in progress. The details and documentation remain to be improved, and the code needs to be cleared and modified. Please let me know if you have any suggestions or questions.

<!---
This file contains the required functions to realize the reconstruction algorithms.
#### 1. Parallel reconstruction algorithms implementation
[//]: # (* gs.py<br/>)
[//]: # (* hio.py<br/>)
[//]: # (* asr.py<br/>)
[//]: # (* mace.py<br/>)
* WF.py<br/>
* pmace.py<br/>
#### 2. Other self defined functions
[//]: # (* error_metrics.py<br/>)
[//]: # (* projections<br/>)
* utils.py<br/>
* preprocess.py<br/>
-->
<!---
## /data
# This directory contains ground truth image used as simulation phantom.
c
[//]: # (## /bm3d-3.0.6)
<!---
This directory contains BM3D python software. For more information, please view [here (http://www.cs.tut.fi/~foi/GCF-BM3D/).
-->

<!---
## main.py
This is the mainfile that integrates the forward model with the reconstruction algorithms. In the forward model, users can self-define the number of [//]: # forward agents and the variance of AWGN. 
-->

<!---
The second part of mainfile.m include the parallel reconstruction algorithms. 
-->

<!---
[//]: # (## Reference)
[//]: # (## License)
## Maintainer
Qiuchen Zhai
-->
