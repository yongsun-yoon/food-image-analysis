import numpy as np
from PIL import Image
from einops import rearrange, repeat

import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size


def owlvit_image_embedder(detector, pixel_values):
    image_embeds = detector.owlvit.vision_model(pixel_values, use_hidden_state=False).last_hidden_state
    image_embeds = detector.owlvit.vision_model.post_layernorm(image_embeds)

    class_token_out = repeat(image_embeds[:, 0, :], 'B D -> B S D', S=image_embeds.size(1) - 1)
    image_embeds = image_embeds[:, 1:, :] * class_token_out
    image_embeds = detector.layer_norm(image_embeds) # (batch, patches*patches, dim)
    image_embeds_2d = rearrange(image_embeds, 'B (H W) D -> B H W D', H=int(np.sqrt(image_embeds.size(1))))

    image_class_embeds = detector.class_head.dense0(image_embeds)
    image_class_embeds = image_class_embeds / torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6
    pred_boxes = detector.box_predictor(image_embeds, image_embeds_2d) # cxcywh
    return image_embeds, image_class_embeds, pred_boxes


def owlvit_text_embedder(detector, input_ids, attention_mask):
    text_embeds = detector.owlvit.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
    text_embeds = detector.owlvit.text_projection(text_embeds)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    return text_embeds


def owlvit_class_predictor(detector, image_embeds, image_class_embeds, text_embeds):    
    pred_logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, text_embeds)
    logit_shift = detector.class_head.logit_shift(image_embeds)
    logit_scale = detector.class_head.logit_scale(image_embeds)
    logit_scale = detector.class_head.elu(logit_scale) + 1
    pred_logits = (pred_logits + logit_shift) * logit_scale
    return pred_logits.squeeze(-1).sigmoid()


def cxcywh_to_xyxy(x, scale=1.):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    xyxy = torch.stack(b, dim=-1)
    return torch.clamp(xyxy, min=0., max=1.) * scale


DEVICE = 'cuda:2'
MODEL_NAME = 'google/owlvit-base-patch32'
DATADIR = '/workspace/mydata/media/'
THRESHOLD = 0.05
CLASS_NAMES = ['food']

processor = OwlViTProcessor.from_pretrained(MODEL_NAME)
detector = OwlViTForObjectDetection.from_pretrained(MODEL_NAME)
_ = detector.eval().requires_grad_(False).to(DEVICE)

text_inputs = processor(text=CLASS_NAMES, return_tensors='pt').to(DEVICE)
text_embeds = owlvit_text_embedder(detector, **text_inputs)


class Model(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.detector = detector
        self.text_embeds = text_embeds

    def predict(self, tasks, **kwargs):
        total_predictions = []
        for task in tasks:
            image_url = task['data']['image']
            image_url = '/'.join(image_url.split('/')[2:])
            image_path = f'{DATADIR}/{image_url}'
            width, height = get_image_size(image_path)
            image = Image.open(image_path).convert('RGB')
            
            image_inputs = self.processor(images=image, return_tensors='pt').to(DEVICE)
            image_embeds, image_class_embeds, pred_boxes = owlvit_image_embedder(self.detector, **image_inputs)
            pred_scores = owlvit_class_predictor(self.detector, image_embeds, image_class_embeds, self.text_embeds)
            pred_boxes[:, :, 0] = pred_boxes[:, :, 0] - pred_boxes[:, :, 2] * 0.5
            pred_boxes[:, :, 1] = pred_boxes[:, :, 1] - pred_boxes[:, :, 3] * 0.5
            pred_boxes = pred_boxes * 100.

            idxs1, idxs2 = torch.where(pred_scores > THRESHOLD)
            cand_boxes = pred_boxes[idxs1, idxs2]
            cand_scores = pred_scores[idxs1, idxs2]
            cand_boxes = cand_boxes.cpu().numpy()
            cand_scores = cand_scores.cpu().numpy()
            
            predictions = []
            scores = []
            for bbox, score in zip(cand_boxes, cand_scores):

                predictions.append({
                    'from_name': 'label',
                    'to_name': 'image',
                    'type': 'rectanglelabels',
                    'scores': float(score),
                    'value': {
                        'rectanglelabels': ['food'],
                        # 'x': float(bbox[0]),
                        # 'y': float(bbox[1]),
                        # 'width': float(bbox[2] - bbox[0]),
                        # 'height': float(bbox[3] - bbox[1]),
                        'x': float(bbox[0]),
                        'y': float(bbox[1]),
                        'width': float(bbox[2]),
                        'height': float(bbox[3]),
                    }
                })               
                scores.append(score)
            
            scores = sum(scores) / max(len(scores), 1)
            total_predictions.append({
                'result': predictions,
                'score': scores
            })
            
        return total_predictions






