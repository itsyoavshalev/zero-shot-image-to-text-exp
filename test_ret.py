import numpy as np
import torch
import cv2

int_mode = 'bilinear'

def show_image_relevance(image_relevance, image, device):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode=int_mode)
    image_relevance = image_relevance.reshape(224, 224).to(device).data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

    return vis, image_relevance

# #https://github.com/openai/CLIP/issues/57
# def convert_models_to_fp32(model): 
#     for p in model.parameters(): 
#         p.data = p.data.float() 
#         if p.grad != None:
#             p.grad.data = p.grad.data.float() 

def interpret(logits_per_text, batch_size, model, device, detach):
    start_layer = 11
    
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_text.shape[0], logits_per_text.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_text.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.to(device) * logits_per_text)
    # model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0]
        cam = blk.attn_probs
        if detach:
            grad = grad.detach()
            cam = cam.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]

    return image_relevance

def generate_heatmap(logits_per_text, batch_size, model, device, detach):
    raw_image_relevance = interpret(logits_per_text, batch_size, model, device, detach)
    with torch.no_grad():
        image_relevance = raw_image_relevance.reshape(batch_size, 1, 7, 7)
        image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode=int_mode)

        min_v = image_relevance.view(batch_size, -1).min(-1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        max_v = image_relevance.view(batch_size, -1).max(-1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)

        # tshape = image_relevance.shape
        # image_relevance = image_relevance.view(tshape[0],-1).softmax(-1)
        # norm_image_relevance = image_relevance.view(tshape)
        norm_image_relevance = (image_relevance - min_v) / (max_v - min_v)

    return norm_image_relevance, raw_image_relevance