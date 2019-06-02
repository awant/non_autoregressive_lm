# ParalARAE

An implementation of Non-autoregressive LM. ARAE model was used as a base ("Adversarially Regularized Autoencoders (ICML 2018)")

ARAE model was taken from https://github.com/awant/arae

To evaluate a model you can download a pretrained kenlm model (the model trained on the same train.txt file):

* SNLI model: https://drive.google.com/file/d/1zpVYD9USw8fxrTl1VzSTKpBHgUaUr5XW/view?usp=sharing

### Training:

dec_type:

* lstm: autoregressive model (arae setting)
* dense: parallel decoding from internal representation on constant positions
* dense_pos: the same as dense, but with positional encoding
* conv: usage of convolutional layers

```console
python train.py --data data_snli --no_earlystopping --gpu 0 --kenlm_model knlm_snli.arpa --dec_type dense
```

##### Additional options:

| option             | description                                             |
|--------------------|---------------------------------------------------------|
| --tensorboard      | draw graphs. need tensorboardx to work                  |
| --kenlm_model      | path to reference kenlm model for computing forward ppl |
| --gpu              |  -1 - don't use gpu, > -1 - use                         |
| --compressing_rate | -S param for kenlm cmd line util                        |


### Generating sentences:

```console
python generate.py --greedy
```

Presentation: https://github.com/awant/non_autoregressive_lm/blob/master/ParallARAE.pdf
