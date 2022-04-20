---
finished_date: 2021-12-14
tags:
    - OpenCV
    - Numpy
    - Python
    - image_descriptors
    - BOW
    - K-Mean
    - sikit_learn
---
# OpenCV_implementation4
- 1000 images are used which are from UKBench dataset
- for each image, 4 relative images are given
- for each image, 128 dimension vector is given as SIFT feature as binary file
- visual words are computed with K-Mean clustring by sikit-learn
- by finding closest centroid, map iamge features to visual word.

## NOTICE
- *Eval.exe* used to show accuracy result is not implemented by myself.

## Visual Dictionary
- *keman_cv4.cpy*
- vertically stack all SIFT descriptors to make global visual dictionary
- find 128 cluster's centroid by KMeans
- Google colab is used
- save the result as *codeword_center.npy*

## Encoding
- for each image, count occurences for each visual word 
- by Term Frequency Inverse Document Frequency(TF IDF), get weighted descriptor
- store the result as *image_descriptor.des*

- more detail is included in the [document](https://github.com/dongwon18/OpenCV_implementation4/blob/main/image_descriptor_document.pdf)

## File structure
```
|-- src
    |-- codeword_center.py
    |-- compute_descriptors.py
    |-- kmean_cv4.py
|-- result
    |-- accuracy.jpg
    |-- image_descriptor.des
|-- image_descriptor_document.pdf
```

## Result
<p align=center>
    <img src="./result/accuracy.jpg" alt="accuracy of descriptor"><br/>
    accuracy
</p>

## 배운 점
- image descriptor를 찾는 방식 중 BOW 방식을 이해하고 구현하였다.
- sikit learn을 이용하여 보다 쉽게 Kmean을 구할 수 있었다.
- Numpy 사용이 능숙해졌다.

## 한계점
- visual word의 개수가 많을수록 정확도는 높아지나 Kmean clustering 진행 시 K의 크기가 커질수록 많은 시간이 걸린다. 또한 가장 가까운 centroid를 찾을 때 걸리는 시간 또한 visual word의 개수에 비례하여 증가하기 때문에 visual word의 개수만을 키우는 것에는 한게가 있다.
- hierarchical Kmean, tree 구조 등 Kmean을 근사하여 찾는 알고리즘과 가장 가까운 centroid를 근사하여 찾는 방법을 통해 정확도와 실행 시간에 있어 이득을 얻을 수 있다.
- 그럼에도 불구하고 BOW 방식을 큰 데이터에 사용하는데에는 한계가 있다.
