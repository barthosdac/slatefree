# A Modified Version of RecSim
This is a modified version of the [RecSim](https://github.com/google-research/recsim) library provided by Google.

The main contributions are as follows:

## Single-User Experiments
In the RecSim library, a different user is generated for each episode of the experiment. Here, there is an option to retain the same user throughout an experiment.

## Learning of the $u$ Vector
RecSim only considers the warm-starting case where the latent representation of a user is directly observable. Here, the possibility of learning this vector at the beginning of each experiment is implemented.

## Addition of slateFree
The SlateFree agent is implemented.

# Repository Contents
The modified version of the RecSim library is contained in the "recsim_mod" directory.


A more detailed description of the modifications can be found in the three markdown files in this repository.

In the "examples" directory, there is an execution example of .py generating data for:
- slateFree(SUM) in a single-user scenario where the user vector is directly readable by the agent.
- slateQ in a multi-user scenario where the user vector is learned over the first 15 episodes.

Additionaly there is .ipynb files to explain how to draw graphs from the data.

There also is a notebook for an exemple of tabular exemples.