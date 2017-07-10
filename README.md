# dark-visdom
Darknet wrapper in Python with Visdom visualization

- Works with Python 2.7.
- Download and build Darknet first.
- Use GPU flag in Darknet `Makefile` to get faster training:

```Makefile
GPU=1
CUDNN=1
OPENCV=0
DEBUG=1

ARCH= -gencode arch=compute_61,code=[sm_61,compute_61]
```

CUDNN is faster but uses more memory on your GPU. OpenCV not really required for training. Debug in case Darknet dumps core.

## To do
- [x] Python wrapper for Darknet training using `subprocess.Popen`
- [x] Log output from Darknet stdout
- [x] Plot learning curves with Visdom
- [ ] Implement early stopping
