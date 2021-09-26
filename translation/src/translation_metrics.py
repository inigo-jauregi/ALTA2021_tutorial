import torch


def translation_accuracy(out_logits, labels, tokenizer=None):

    # Softmax
    out_logits = out_logits.softmax(dim=-1).argmax(dim=-1)
    # View each token
    out_logits_reshaped = out_logits.view(-1, 1)
    # Correct token level predictions
    equals_vec = out_logits_reshaped.eq(labels.view(-1, 1))
    # Accuracy
    acc = torch.sum(equals_vec) / equals_vec.size()[0]

    return acc
