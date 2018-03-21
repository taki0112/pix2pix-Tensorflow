# pix2pix-Tensorflow
SImple Tensorflow implementations of " Image-to-Image Translation with Conditional Adversarial Networks" (CVPR 2017)

## Requirements
* Tensorflow 1.4
* Python 3.6

## Usage
```bash
├── dataset
   └── YOUR_DATASET_NAME
       ├── trainA
           ├── 1.jpg (format doesn't matter)
           ├── 2.png
           └── ...
       ├── trainB (target list)
           ├── 1_.jpg
           ├── 2_.png
           └── ...
       ├── testA
           ├── aaa.jpg 
           ├── bbb.png
           └── ...

```

```bash
> python main.py
```


## Author
Junho Kim
