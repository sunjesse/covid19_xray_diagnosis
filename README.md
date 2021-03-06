# covid19_xray_diagnosis
PyTorch code template for those who want to cure coronavirus from x-ray scans.

We greatly thank Wang et al. 2020 for the preprocessed data. Please visit their repository and cite their work if you find the data or their method helpful: https://github.com/lindawangg/COVID-Net.
We just wrote this script and pipelined everything together in PyTorch for PyTorch users.

The default code uses a DenseNet-121 and gets around 78% accuracy. Feel free to play around with different methods!

Running the code is simple. First download the .npy data from Wang et al.'s repository. Then fill in the respective paths in train.py.
Once that is done, as long as you have a GPU (get rid of the .cuda() substrings in the code if you don't), type the following in commandline:

```
python3 train.py --lr 0.000001 --optimizer adam --batch_size 16 --epoch 100
```

If anything doesn't work, just leave an issue. I wrote this script up in less than an hour so there's bound to be error.

Together with the power of deep learning we can cure the corona!

![alt text](https://github.com/sunjesse/covid19_xray_diagnosis/blob/master/wat.png)
