# semi-global-matching
An implementation of Semi-Global Matching (SGM) on CPU

## Description
- An implementation of Semi-Global Matching (SGM) based on [1].
- The 9x7 Census transform is used as matching cost.
- The following has not been implemented.
  - median filtering
  - consistency check

## References
- [1] Hirschmüller, H., Accurate and Efficient Stereo Processing by Semi-Global Matching and Mutual Information, IEEE Conference on Computer Vision and Pattern Recognition, June 2005

## Requirement
- OpenCV
- AVX2
- OpenMP (optional)

## How to build
```
$ git clone https://github.com/gishi523/semi-global-matching.git
$ cd semi-global-matching
$ mkdir build
$ cd build
$ cmake ../
$ make
```

## How to run
```
./sgm left-image-format right-image-format
```
- left-image-format
    - the left image sequence
- right-image-format
    - the right image sequence

### Example
 ```
./sgm images/img_c0_%09d.pgm images/img_c1_%09d.pgm
```

## Author
gishi523