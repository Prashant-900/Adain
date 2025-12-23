import io
from typing import Optional, List

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from PIL import Image
from torchvision.transforms import functional as F
import torch.nn.functional as tF

from architecture import vgg, decoder, device
from scripts import style_transfer, test_transform

app = FastAPI(title="AdaIN Style Transfer Server")


def _load_image(file_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


def _preprocess(img: Image.Image, size: int) -> torch.Tensor:
    tf = test_transform(size=size)
    tensor = tf(img)
    return tensor.unsqueeze(0)


def _postprocess(tensor: torch.Tensor) -> bytes:
    tensor = torch.clamp(tensor, 0.0, 1.0).squeeze(0).cpu()
    pil_img = F.to_pil_image(tensor)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()

def _preprocess_style_tensor(t: torch.Tensor, size: int) -> torch.Tensor:
    _, _, h, w = t.shape
    if size and size > 0:
        scale = size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        t = tF.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return t


def _load_mask(file_bytes: bytes) -> torch.Tensor:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("L")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid mask image: {e}")
    arr = F.pil_to_tensor(img).float() / 255.0
    return (arr > 0.5).float()


@app.post("/stylize", response_description="Stylized image (PNG)")
async def stylize(
    content: UploadFile = File(...),
    style: List[UploadFile] = File(...),
    mask: Optional[List[UploadFile]] = File(None),
    alpha: float = Query(1.0, ge=0.0, le=1.0),
    size: int = Query(0, ge=0),
    preserve_color: bool = Query(False),
    weights: Optional[str] = Query(None),
):
    try:
        content_bytes = await content.read()
        style_bytes_list = [await s.read() for s in style]
        mask_bytes_list = [await m.read() for m in mask] if mask is not None else None
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read files: {e}")

    content_img = _load_image(content_bytes)
    content_t = _preprocess(content_img, size).to(device)

    style_tensors: List[torch.Tensor] = []
    for sb in style_bytes_list:
        s_img = _load_image(sb)
        s_t = _preprocess(s_img, size).to(device)
        style_tensors.append(s_t)

    if len(style_tensors) > 1:
        style_tensors = [
            _preprocess_style_tensor(s_t, size) for s_t in style_tensors
        ]
        target_h, target_w = style_tensors[0].shape[-2], style_tensors[0].shape[-1]
        style_tensors = [
            s_t if s_t.shape[-2:] == (target_h, target_w)
            else tF.interpolate(s_t, size=(target_h, target_w), mode="bilinear", align_corners=False)
            for s_t in style_tensors
        ]

    mask_t = None
    if mask_bytes_list is not None:
        mask_tensors: List[torch.Tensor] = []
        for mb in mask_bytes_list:
            m = _load_mask(mb)
            mask_tensors.append(m.to(device))
        if len(mask_tensors) == 1:
            mask_t = mask_tensors[0]
        else:
            mask_t = mask_tensors

    if len(style_tensors) == 1:
        style_arg = style_tensors[0]
    else:
        style_arg = style_tensors

    style_interp_weights = None
    if weights is not None:
        try:
            style_interp_weights = [float(x) for x in weights.split(",") if x.strip() != ""]
            if isinstance(style_arg, list) and len(style_interp_weights) != len(style_arg):
                raise ValueError("weights length must match number of style images")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid weights: {e}")

    if mask_t is not None and not isinstance(mask_t, list):
        if not isinstance(style_arg, list) or len(style_arg) != 2:
            raise HTTPException(status_code=400, detail="single mask requires exactly two style images for FG/BG")
    
    if mask_t is not None and isinstance(mask_t, list):
        if not isinstance(style_arg, list) or len(mask_t) != len(style_arg):
            raise HTTPException(status_code=400, detail="number of masks must match number of styles for multi-mask mode")

    with torch.no_grad():
        output = style_transfer(
            vgg,
            decoder,
            content_t,
            style_arg,
            alpha=alpha,
            mask=mask_t,
            preserve_color=preserve_color,
            style_interp_weights=style_interp_weights,
        )

    img_bytes = _postprocess(output)
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")


@app.get("/health")
def health():
    return {"status": "ok"}
