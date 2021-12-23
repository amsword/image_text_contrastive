import torch


def all_gather_grad(x):
    all_x = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(all_x, x)
    all_x[torch.distributed.get_rank()] = x # essential to propagate gradient on x
    x = torch.cat(all_x, dim=0)
    return x

def image_text_contrastive_loss(image_feat, text_feat, temperature, image_id, text_id):
    image_feat = torch.nn.functional.normalize(image_feat, dim=1)
    text_feat = torch.nn.functional.normalize(text_feat, dim=1)

    # add the following 4 lines
    image_feat = all_gather_grad(image_feat)
    text_feat = all_gather_grad(text_feat)

    image_id = all_gather_grad(image_id)
    text_id = all_gather_grad(text_id)

    logits = torch.matmul(image_feat, text_feat.t())
    logits /= temperature

    gt_image = image_id.reshape((-1, 1)) == image_id.reshape((1, -1))
    gt_text = text_id.reshape((-1, 1)) == text_id.reshape((1, -1))
    gt = torch.logical_or(gt_image, gt_text)

    loss1 = -torch.sum(gt * torch.nn.functional.log_softmax(logits, dim=1)) / gt.sum()
    loss2 = -torch.sum(gt.t() * torch.nn.functional.log_softmax(logits.t(), dim=1)) / gt.sum()
    return (loss1 + loss2) / 2 * torch.distributed.get_world_size() # scale it up by the number of GPUs
