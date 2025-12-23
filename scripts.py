import torch
import torch.nn.functional as F
from torchvision import transforms
from utils import adaptive_instance_normalization

def _flatten_hw(x):
    n, c, h, w = x.size()
    return x.view(n, c, h * w)


def _coral_tensor(src, ref, eps=1e-5):
    assert src.size(1) == 3 and ref.size(1) == 3
    src = src.clone()
    ref = ref.clone()
    n_s, c, h_s, w_s = src.size()
    n_r, _, h_r, w_r = ref.size()
    src_f = src.view(n_s, c, -1)
    ref_f = ref.view(n_r, c, -1)
    src_mean = src_f.mean(dim=2, keepdim=True)
    ref_mean = ref_f.mean(dim=2, keepdim=True)
    src_center = src_f - src_mean
    ref_center = ref_f - ref_mean
    m_s = src_center.size(2)
    m_r = ref_center.size(2)
    cov_s = torch.bmm(src_center, src_center.transpose(1, 2)) / (m_s - 1)
    cov_r = torch.bmm(ref_center, ref_center.transpose(1, 2)) / (m_r - 1)
    eye = torch.eye(c, device=src.device).unsqueeze(0).expand_as(cov_s)
    cov_s = cov_s + eps * eye
    cov_r = cov_r + eps * eye
    vals_s, vecs_s = torch.linalg.eigh(cov_s)
    vals_r, vecs_r = torch.linalg.eigh(cov_r)
    sqrt_s = vecs_s @ torch.diag_embed(torch.sqrt(torch.clamp(vals_s, min=0.0))) @ vecs_s.transpose(1, 2)
    invsqrt_s = vecs_s @ torch.diag_embed(1.0 / torch.sqrt(torch.clamp(vals_s, min=eps))) @ vecs_s.transpose(1, 2)
    sqrt_r = vecs_r @ torch.diag_embed(torch.sqrt(torch.clamp(vals_r, min=0.0))) @ vecs_r.transpose(1, 2)
    a = sqrt_r @ invsqrt_s
    transformed = a @ src_center + ref_mean
    out = transformed.view(n_s, c, h_s, w_s)
    return torch.clamp(out, 0.0, 1.0)


def style_transfer(vgg, decoder, content, style, alpha=1.0, mask=None, preserve_color=False, style_interp_weights=None):
    assert (0.0 <= alpha <= 1.0)
    if preserve_color:
        if isinstance(style, (list, tuple)):
            style = [
                _coral_tensor(s, content) for s in style
            ]
        else:
            style = _coral_tensor(style, content)

    content_f = vgg(content)

    if isinstance(style, (list, tuple)):
        style_batch = torch.cat(style, dim=0)
    else:
        style_batch = style

    style_f = vgg(style_batch)

    if mask is not None:
        _, c, h, w = content_f.size()
        if isinstance(mask, (list, tuple)):
            assert len(mask) == style_f.size(0), "Number of masks must match number of styles"
            target_feat = torch.zeros_like(content_f)
            total_weight = torch.zeros(1, 1, h, w, device=content_f.device)
            for i, m in enumerate(mask):
                if m.dim() == 2:
                    m = m.unsqueeze(0).unsqueeze(0)
                elif m.dim() == 3:
                    m = m.unsqueeze(0)
                m_resized = F.interpolate(m.float(), size=(h, w), mode="bilinear", align_corners=False)
                m_resized = torch.clamp(m_resized, 0.0, 1.0)
                style_feat_i = style_f[i].unsqueeze(0)
                t_i = adaptive_instance_normalization(content_f, style_feat_i)
                target_feat = target_feat + t_i * m_resized
                total_weight = total_weight + m_resized
            total_weight = torch.clamp(total_weight, min=1e-8)
            target_feat = target_feat / total_weight
            unmask_weight = (total_weight < 0.01).float()
            target_feat = target_feat * (1 - unmask_weight) + content_f * unmask_weight
            feat = target_feat * alpha + content_f * (1 - alpha)
            return decoder(feat)
        else:
            assert style_f.size(0) == 2
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(0)
            mask_resized = F.interpolate(mask.float(), size=(h, w), mode="nearest")
            mask_flat = mask_resized.view(1, -1).squeeze(0) > 0.5
            content_vec = content_f.view(c, -1)
            idx_fg = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
            idx_bg = torch.nonzero(~mask_flat, as_tuple=False).squeeze(1)
            target_vec = content_vec.clone()
            if idx_fg.numel() > 0:
                content_fg = content_vec.index_select(1, idx_fg).view(1, c, -1, 1)
                style_fg = style_f[0].unsqueeze(0)
                t_fg = adaptive_instance_normalization(content_fg, style_fg).view(c, -1)
                target_vec[:, idx_fg] = t_fg
            if idx_bg.numel() > 0:
                content_bg = content_vec.index_select(1, idx_bg).view(1, c, -1, 1)
                style_bg = style_f[1].unsqueeze(0)
                t_bg = adaptive_instance_normalization(content_bg, style_bg).view(c, -1)
                target_vec[:, idx_bg] = t_bg
            feat = target_vec.view(1, c, h, w)
            feat = feat * alpha + content_f * (1 - alpha)
            return decoder(feat)

    if style_interp_weights is not None and style_f.size(0) > 1:
        weights = torch.tensor(style_interp_weights, dtype=content_f.dtype, device=content_f.device)
        weights = weights / (weights.sum() + 1e-8)
        t_sum = torch.zeros_like(content_f)
        for i in range(style_f.size(0)):
            t_i = adaptive_instance_normalization(content_f, style_f[i].unsqueeze(0))
            t_sum = t_sum + weights[i] * t_i
        feat = t_sum * alpha + content_f * (1 - alpha)
        return decoder(feat)

    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

def test_transform(size=512):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

content_tf = test_transform()
style_tf = test_transform()