# NTHU_LAB-Model-Compression-MESA-2

## 流程概覽

### Initial Train
* 一般的訓練

### Prune
* 採用 norm-base 方法對 convolution layers 每一層中的 **filters** 進行裁剪。
以 Alexnet 為例，有 5 層卷積層，參考 [Deep Compression](https://arxiv.org/pdf/1510.00149.pdf) 與 [Pruning Filters For Efficient Convnets](https://arxiv.org/pdf/1608.08710.pdf) 的做法，對每一層**filters** 的裁剪率分別為 `0.16, 0.62, 0.65, 0.63, 0.63`。

![](https://i.imgur.com/ypr8RdG.png)

### Prune Retrain
* 幾乎如同一般的訓練，差別在
    * 將被 prune 掉的 nodes 的 gradient 為 0，確保他們不被更新
    * 除了原本對 label 的 loss ，加入 ==convolution layers loss==:
    
![](https://i.imgur.com/uAjGo8v.jpg)


### Quantize
* 參考 [Deep Compression](https://arxiv.org/pdf/1510.00149.pdf)，對每一層 convolution layers 進行 quantize
```python=
for layer in conv_layers:
    1. 將沒被 prune 掉的 filters weights 展開成一維 flat_arr
    2. 利用 Kmean 對 flat_arr 分成 32 (5 bits) 類
    3. 將原本 layer 中被歸為同一類的，替換成該類的centroid，因此替換後整
       層 layer 的 weights 只會由 32 個 value 表示
```

### Quantize Retrain
* 幾乎如同一般的訓練，差別在
    * 確保被 prune 掉的 nodes 的 gradient 為 0，讓他們不被更新
    * 被 quantize 為同一類的 weights，將他們個別的 gradient 替換為同一類的 weights 的 gradient 的 mean，以讓他們在更新後保持一樣的值
