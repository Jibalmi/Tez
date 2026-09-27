"""Full-vocabulary letter readout with one independent option mask per decision."""


def option_logp(last_logits, letter_ids, counts):
    import torch
    import torch.nn.functional as F
    if (last_logits.ndim != 2 or letter_ids.ndim != 1 or len(counts) != last_logits.shape[0]
            or not counts or any(type(k) is not int or not 2 <= k <= len(letter_ids) for k in counts)):
        raise ValueError("Require one valid option count for each vocabulary-logit row")
    maximum = max(counts)
    selected = last_logits.float()[:, letter_ids[:maximum]]
    mask = torch.arange(maximum, device=selected.device)[None, :] < torch.tensor(counts, device=selected.device)[:, None]
    return F.log_softmax(torch.where(mask, selected, torch.full_like(selected, -1e4)), dim=-1)
