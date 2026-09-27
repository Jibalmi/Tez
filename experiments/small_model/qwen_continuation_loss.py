"""Original teacher KL and gold CE for a single unpadded continuation decision."""


def decision_loss(logits, target, gold=None):
    """Caller divides each loss by the complete effective batch size before backward."""
    import torch
    import torch.nn.functional as F
    if logits.ndim != 1 or target.shape != logits.shape or len(logits) < 2:
        raise ValueError("Require matching unpadded option vectors")
    if target.requires_grad or not target.is_floating_point():
        raise ValueError("Teacher probabilities must be fixed floating point targets")
    if (not torch.isfinite(target).all().item() or (target < 0).any().item()
            or not torch.isclose(target.sum(), target.new_tensor(1.0), atol=1e-5, rtol=0).item()):
        raise ValueError("Require a finite normalized teacher distribution")
    if gold is not None and (type(gold) is not int or not 0 <= gold < len(logits)):
        raise ValueError("Gold index must identify a supplied option")
    logp = F.log_softmax(logits.float(), dim=-1)
    teacher = target.float()
    loss = (teacher * (teacher.clamp_min(1e-12).log() - logp)).sum()
    if gold is not None:
        loss = loss - .5 * logp[gold]
    return loss
