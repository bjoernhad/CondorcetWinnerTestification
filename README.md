## Code Repository for the Paper
# "Testification of Condorcet Winners in Dueling Bandits"
## UAI 2021

The code is written in Python 3.7.

You can cite our paper as follows:

```
@inproceedings{Haddenhorst2021,
  title={Testification of Condorcet Winners in Dueling Bandits},
  author={Haddenhorst, Bj{\"o}rn and Bengs, Viktor and Brandt, Jasmin and H{\"u}llermeier, Eyke},
  booktitle = {Proceedings of Conference on Uncertainty in Artificial Intelligence (UAI)},
  year={2021},
}
```

## Requirements
To install the requirements:

```setup
pip install -r requirements.txt
```

## Evaluation
The results from our paper have been generated with

- Experiments_Passive.py for the experiment in Section 7.2,
- Experiments_Active.py for the comparison of SELECT-then-Verify and Noisy Tournament Sampling.

For reconstructing the results, uncomment the corresponding lines and execute
the files. Some results are not shown in the terminal but only saved to files,
see the code for the details.

In case of questions please contact Björn Haddenhorst (bjoernha@mail.upb.de).
