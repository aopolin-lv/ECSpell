# ECSpell
Code for paper "General and Domain Adaptive Chinese Spelling Check with Error Consistent Pretraining"
## Data usage
Path: `Data/domains_data`
- For zero-shot tasks, you should combine the *.train file and *.test file.
- For common tasks, the *.train file is used to do training and do evaluating while *.test is adopted to do predicting.
## Usage:
``` shell
cd glyce
python setup.py develop
pip show glyce   # to ensure the successful installation of glyce lib
```
[Model weights](https://drive.google.com/file/d/1HlfDbMpXR6YHiBuJS8s_K3ZKG6j0fvc5/view?usp=sharing)
## Citation
```
@article{lv2023general,
  title={General and Domain-adaptive Chinese Spelling Check with Error-consistent Pretraining},
  author={Lv, Qi and Cao, Ziqiang and Geng, Lei and Ai, Chunhui and Yan, Xu and Fu, Guohong},
  journal={ACM Transactions on Asian and Low-Resource Language Information Processing},
  volume={22},
  number={5},
  pages={1--18},
  year={2023},
  publisher={ACM New York, NY}
}
```
