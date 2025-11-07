import torch

def make_src_mask(src, src_pad_idx):
    """
    src: [batch size, src len]
    return: [batch size, 1, 1, src len]
    """
    # 保证mask类型为bool并与src同设备
    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)
    src_mask = src_mask.to(dtype=torch.bool, device=src.device)
    return src_mask


def make_trg_mask(trg, trg_pad_idx, device=None):
    """
    trg: [batch size, trg len]
    return: [batch size, 1, trg len, trg len]
    """
    trg_device = trg.device if device is None else device

    # pad mask
    trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(2)
    trg_pad_mask = trg_pad_mask.to(dtype=torch.bool, device=trg_device)

    # subsequent mask (防止看到未来词)
    trg_len = trg.shape[1]
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg_device)).bool()

    # combine
    trg_mask = trg_pad_mask & trg_sub_mask
    trg_mask = trg_mask.to(dtype=torch.bool, device=trg_device)

    return trg_mask
