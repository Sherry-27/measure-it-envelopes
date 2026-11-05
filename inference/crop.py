"""Refine an automatically detected envelope without changing camera coordinates."""
import cv2
import numpy as np

from models.network import tensor_image, working_frame


def refine_mask(frame, coarse_mask, model, device, threshold):
    ys, xs = np.nonzero(coarse_mask)
    if len(xs) == 0:
        return None, None, "crop_object_not_found", None
    # Fixed before validation: 20% on each side, with a 16 px minimum.
    pad_x=max(16,int(np.ceil((xs.max()-xs.min()+1)*0.2)))
    pad_y=max(16,int(np.ceil((ys.max()-ys.min()+1)*0.2)))
    x0=max(0,int(xs.min())-pad_x); x1=min(frame.shape[1],int(xs.max())+1+pad_x)
    y0=max(0,int(ys.min())-pad_y); y1=min(frame.shape[0],int(ys.max())+1+pad_y)
    bounds=[x0,y0,x1,y1]
    crop=working_frame(frame[y0:y1,x0:x1])
    pred=model([tensor_image(crop).to(device)])[0]
    score=float(pred["scores"][0]) if len(pred["scores"]) else None
    if score is None or score < threshold:
        return None,score,"crop_object_not_found",bounds
    if len(pred["scores"])>1 and float(pred["scores"][1])>=threshold:
        return None,score,"crop_multiple_objects",bounds
    binary=(pred["masks"][0,0].cpu().numpy()>=0.5).astype(np.uint8)
    restored=cv2.resize(binary,(x1-x0,y1-y0),interpolation=cv2.INTER_NEAREST)
    if restored[0].any() or restored[-1].any() or restored[:,0].any() or restored[:,-1].any():
        return None,score,"crop_mask_touches_edge",bounds
    full=np.zeros(frame.shape[:2],np.uint8)
    full[y0:y1,x0:x1]=restored
    return full,score,None,bounds

