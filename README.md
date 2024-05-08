# RangeNet Semantic Segmentation

## Source of the model

	  Model picked up from 'https://github.com/PRBonn/lidar-bonnetal/tree/master'

---

## Description of the model

	> Model     : RangeNet
    > Backbone  : Squeezeseg
	> Input size: [1, 5, 64, 2048]

---

## Framework and version

    AIMET   : torch-gpu-1.24.0
    offset  : 11
    pytorch : 1.9.1+cu111
    python  : 3.8

---

## Modifications done to the model (if any)

In src/backbone/squeezeseg.py changed forward function:
```python
    def run_layer1(self, x, layer, skips, os):
        y = layer(x)
        skips[os] = x.detach()
        os *= 2
        x = y
        return x, skips, os

    def run_layer2(self, x, layer, skips, os):
        y = layer(x)
        x = y
        return x, skips, os

    def forward(self, x):
        # filter input
        x = x[:, self.input_idxs]

        # run cnn
        # store for skip connections
        skips = {}
        os = 1

        # encoder
        skip_in = self.conv1b(x)
        x = self.conv1a(x)
        # first skip done manually
        skips[1] = skip_in.detach()
        os *= 2

        x, skips, os = self.run_layer1(x, self.fire23, skips, os)
        x, skips, os = self.run_layer2(x, self.dropout, skips, os)
        x, skips, os = self.run_layer1(x, self.fire45, skips, os)
        x, skips, os = self.run_layer2(x, self.dropout, skips, os)
        x, skips, os = self.run_layer1(x, self.fire6789, skips, os)
        x, skips, os = self.run_layer2(x, self.dropout, skips, os)

        return x, skips
```

In src/decoders/squeezeseg.py changed run_layer function:
```python
def run_layer(self, x, layer, skips, os):
    feats = layer(x)  # up
    # print(feats.shape[-1] ,x.shape[-1])
    # if feats.shape[-1] > x.shape[-1]:
    os //= 2  
    feats = feats + skips[os].detach()  # add skip
    x = feats
    return x, skips, os
```

---

## Execution command
Command to run:
```bash
python src/quant.py --config config/rangenet.json
```

---

## list of operators in this model

 	{'Add', 'Conv', 'Concat', 'Gather',  'MaxPool', 'Relu', 'Softmax', 'Transpose'}

---

## Trained on dataset(s)

Kitti Pretrained

---

<!-- ## Path to datasets

	- Internal Datsets - Used 2k images for calibration and 50k images for validation from Imagenet Dataset.
	- External Datasets - URLs -->

---

## Result:

  - RangeNet Squeezeseg
    - FP32 
        - Acc avg: 0.724
        - IoU avg: 0.266
    - Quantized PTQ
        - Acc avg: 0.526
        - IoU avg: 0.188
    - Quantized QAT
        - Acc avg 0.757
        - IoU avg 0.263
        
---