# image_text_contrastive
image-text contrastive loss implementation

1. Install by
```bash
pip install https://github.com/amsword/image_text_contrastive
```
2. Use it
```python
from image_text_contrastive import image_text_contrastive_loss as itc
itc(image_feat, text_feat, temperature, image_id, text_id)
```

For more details, please check out [here](https://amsword.github.io/How-To-Implement-Image-Text-Contrastive-loss-Correctly-in-Pytorch/)
