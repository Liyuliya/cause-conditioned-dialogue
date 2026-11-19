
import math


def validate_pair(row):
    p = row["prompt_token_ids"]
    c, r = row["chosen_token_ids"], row["rejected_token_ids"]
    if not p or len(c) != 4 or len(r) != 4 or c == r or c[-1] != r[-1]:
        raise ValueError(
            "Expected a nonempty prompt and distinct four-token action+EOS targets"
        )
    if len(p) + 4 > 2048 or any(type(x) is not int or x < 0 for x in p + c + r):
        raise ValueError("Invalid or overlength token sequence")


def pair_inputs(row):
    validate_pair(row)
    p = row["prompt_token_ids"]
    return [p + row["chosen_token_ids"], p + row["rejected_token_ids"]]


def scalar_loss(pc, pr, rc, rr, beta):
    """Independent scalar oracle for tests (softplus of the negative margin)."""
    z = -beta * ((pc - pr) - (rc - rr))
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


def completion_logps(logits, targets):
    """logits: [2,4,vocab], targets: [2,4]; prompt and final unused logit excluded."""
    import torch

    if logits.ndim != 3 or logits.shape[:2] != targets.shape or targets.shape != (2, 4):
        raise ValueError("Unexpected chosen/rejected logits or targets shape")
    # FP32 log-softmax; normalize over the ENTIRE vocabulary, not the eight codes.
    values = (
        logits.float()
        .log_softmax(-1)
        .gather(-1, targets.unsqueeze(-1))
        .squeeze(-1)
        .sum(-1)
    )
    if not torch.isfinite(values).all().item():
        raise ValueError("Nonfinite completion log probability")
    return values


def pair_logps(model, input_ids, targets, full_logits=False):
    """Single pair per microbatch; both sequences have exactly the same length."""
    import torch

    if input_ids.shape[0] != 2 or not torch.equal(input_ids[:, -4:], targets):
        raise ValueError("Pair target alignment changed")
    kwargs = {} if full_logits else {"logits_to_keep": 5}
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
        **kwargs
    )
    expected = input_ids.shape[1] if full_logits else 5
    if output.logits.shape[1] != expected:
        raise ValueError(
            "Model does not implement the expected logits_to_keep behavior"
        )
    # Positions L-5..L-2 predict the four target tokens at L-4..L-1, including EOS.
    return completion_logps(output.logits[:, -5:-1, :], targets)


def dpo_loss(policy, reference, beta):
    import torch
    import torch.nn.functional as F

    if policy.shape != (2,) or reference.shape != (2,) or beta <= 0:
        raise ValueError("Invalid DPO inputs")
    reference = reference.detach()
    z = beta * ((policy[0] - policy[1]) - (reference[0] - reference[1]))
    loss = -F.logsigmoid(z)
    if not torch.isfinite(loss).item():
        raise ValueError("Nonfinite DPO loss")
    return loss
