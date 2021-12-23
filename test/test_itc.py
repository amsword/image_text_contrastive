from image_text_contrastive import image_text_contrastive_loss as itc
import torch


def test_runnable():
    image_feat = torch.zeros((64, 128))
    text_feat = torch.zeros((64, 128))
    itc(image_feat, text_feat, 0.1)

def test_runnable_with_id():
    image_feat = torch.zeros((64, 128))
    text_feat = torch.zeros((64, 128))
    image_id = torch.arange(64)
    text_id = torch.arange(64)
    itc(image_feat, text_feat, 0.1, image_id, text_id)

def test_runnable_gpu():
    image_feat = torch.zeros((64, 128)).cuda()
    text_feat = torch.zeros((64, 128)).cuda()
    itc(image_feat, text_feat, 0.1)

