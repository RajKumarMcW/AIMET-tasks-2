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

	
---

## Execution command
Command to run:
```bash
python src/quant.py --config config/rangenet.json
```

---

## list of operators in this model

 	{'Add', 'Conv', 'Flatten', 'Gemm', 'GlobalAveragePool', 'MaxPool', 'Relu'}

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
        
---