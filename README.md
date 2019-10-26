# Tree Transformer
This is the official implementation of the paper [Tree Transformer: Integrating Tree Structures into Self-Attention](http://arxiv.org/abs/1909.06639). If you use this code or our results in your research, we'd appreciate you cite our paper as following:

```
@article{Wang2019TreeTransformer,
  title={Tree Transformer: Integrating Tree Structures into Self-Attention},
  author={Yau-Shian Wang and Hung-Yi Lee and Yun-Nung Chen},
  journal={arXiv preprint arXiv:1909.06639},
  year={2019}
}
```

## Dependencies

* python3
* pytorch 1.0

We use BERT tokenizer from [PyTorch-Transformers](https://github.com/huggingface/pytorch-transformers) to tokenize words. Please install [PyTorch-Transformers](https://github.com/huggingface/pytorch-transformers) following the instructions of the repository.  


## Training
For grammar induction training:  
```python3 main.py -train -model_dir [model_dir] -num_step 60000```  
The default setting achieves F1 of approximatedly 49.5 on WSJ test set. The training file 'data/train.txt' includes all WSJ data except 'WSJ_22 and WSJ_23'.   

## Evaluation
For grammar induction testing:  
```python3 main.py -test -model_dir [model_dir]```  
The code creates a result directory named model_dir. The result directory includes 'bracket.json' and 'tree.txt'. File 'bracket.json' contains the brackets of trees outputted from the model and they can be used for evaluating F1. The ground truth brackets of testing data can be obtained by using code of [on-lstm](https://github.com/yikangshen/Ordered-Neurons). File 'tree.txt' contains the parse trees. The default testing file 'data/test.txt' contains the tests of wsj_23.   

## Acknowledgements
* Our code is mainly revised from [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html).  
* The code of BERT optimizer is taken from [PyTorch-Transformers](https://github.com/huggingface/pytorch-transformers).  

## Contact
king6101@gmail.com  
