# ASCII-ART

Turn a simple image to text with ascii characters!
Currently the code works a little, I'm still working on it!

## Usage

Install the requirements:

```
  pip install numpy scipy scikit-image
```
Compile with cython:
```
  python setup.py build_ext --i
```
Note: add -I /usr/local/lib/python2.7/site-packages/numpy/core/include(your numpy include path) if needed.

In python:
```python
  import ascii
  ascii.image_to_ascii('monk_1.bmp', 0.3, Rw=48)
```

## Results:

![](https://raw.githubusercontent.com/MacLeek/ascii-art/master/monk_1.bmp)
![](https://raw.githubusercontent.com/MacLeek/ascii-art/master/1.jpeg)
![](https://raw.githubusercontent.com/MacLeek/ascii-art/master/queen.gif)
![](https://raw.githubusercontent.com/MacLeek/ascii-art/master/2.jpeg)

## Reference

[Structure-based ASCII Art](http://www.cse.cuhk.edu.hk/~ttwong/papers/asciiart/asciiart.html)
