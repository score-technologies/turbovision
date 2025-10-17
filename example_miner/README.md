This is an example of a Hugginface Hub Repo for deploying a chute via Turbovision cli

The following 2 files must be present (in their current locations) to deploy the chute successfully (although their content can be modified): 
- miner.py (specify the type of ML model, how any ML models, how they are orchestrated, any and all preprocessing, postprocessing, etc)
- config.yml (specify the machine specs - ie GPU, etc)

Any other files - e.g. dependencies, utils, weights, etc are optional (and left out of this example repo for simplicity) but should be included as needed. 

NOTE: All network-related operations - ie downloading challenge data, downloading model weights - are handled in the chute outside this repo. So all dependencies (ie weight files) must be defined or contained in this repo which is completely open-source