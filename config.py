import torch
from lion_pytorch import Lion
from transformers import get_linear_schedule_with_warmup


def get_model_paths():
    """
    Return the model paths.

    Returns:
        model_paths: model paths
    """

    return [
        "/docile/baselines/NER/auto_find_best_model_res/model",
        "/docile/baselines/NER/auto_find_best_model_res/backup_ours/6_04_ours",
        "/mlcv/WorkingSpace/Personals/baotg/DocILE/trained_model/roberta_base_fgm",
    ]


def get_optimizer(model, lr=1e-5, weight_decay=1e-2, warmup_steps=0, training_step=0):
    """
    Return the optimizer and scheduler.

    Args:
        model: model to be trained
        lr: learning rate
        weight_decay: weight decay
        warmup_steps: warmup steps
        training_step: total training steps

    Returns:
        optimizer: optimizer
        scheduler: scheduler
    """

    optimizer = Lion(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_step
    )

    return (optimizer, scheduler)


def ensemble_output(outputs):
    """
    Return the ensemble result of multiple models.

    Args:
        outputs: list of outputs from multiple models

    Returns:
        predictions: ensemble predictions
        scores: ensemble scores
    """

    predictions = [torch.where(torch.sigmoid(output.logits) > 0.5, 1, 0) for output in outputs]
    predictions = torch.stack(predictions, dim=-2)
    predictions = torch.where(torch.sum(predictions, -1) >= 1, 1, 0)

    scores = torch.mean(
        torch.sigmoid(torch.stack([output.logits for output in outputs], dim=-1)), dim=-1
    )

    return predictions, scores
