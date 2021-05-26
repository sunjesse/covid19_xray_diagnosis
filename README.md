# covid19_xray_diagnosis
PyTorch code template to diagnose COVID-19 from x-ray scans.

We greatly thank Wang et al. 2020 for the preprocessed data. Please visit their repository and cite their work if you find the data or their method helpful: https://github.com/lindawangg/COVID-Net.
We just wrote this script and pipelined everything together in PyTorch for PyTorch users.

The default code uses a DenseNet-121 and gets around 78% accuracy. Feel free to play around with different methods!

Running the code is simple. First download the .npy data from Wang et al.'s repository. Then fill in the respective paths in train.py.
Once that is done, as long as you have a GPU (get rid of the .cuda() substrings in the code if you don't), type the following in commandline:

```
python3 train.py --lr 0.000001 --optimizer adam --batch_size 16 --epoch 100
```
