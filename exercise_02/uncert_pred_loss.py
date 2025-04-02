import torch

def nll_loss(predictions, true_labels, n_labels):
    """ predictions is the means and the log uncertainties"""
    value_preds = predictions[:, :n_labels]
    uncert_preds_log = predictions[:, n_labels:]
    uncert_preds = torch.exp(uncert_preds_log)
    #uncert_preds_log = 0
        
    l = torch.mean((0.5 * ((value_preds-true_labels)/uncert_preds)**2 + uncert_preds_log))
    #çprint(l)
    return l