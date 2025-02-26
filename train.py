import torch
import metrics
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from pprint import pprint


def train(model, optimizer, loss_fn, p, train_loader, val_loader, test_loader):
    
    early_stop = False

    ### Train model 
    for epoch_idx in tqdm(range(p.n_epochs), desc="Training"):

        # Loop through training data
        epoch_loss = 0.0
        model.train()
        for img, labels in tqdm(train_loader, desc="Epoch"):

            # Change device, datatype and rescale 
            img = img.to(p.device, dtype=torch.float32) / 255.0 # [N,C,H,W]
            labels = labels.to(p.device, dtype=torch.float32) / 255.0 # [N,1,H,W]

            optimizer.zero_grad()

            output_logits = model(img) # [N,1,H,W]
            loss = loss_fn(output_logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch {1+epoch_idx} loss = {epoch_loss:.4f}")

        # Validation
        print("Performing validation")
        val_metrics = test(model, p, val_loader)
        pprint(val_metrics)

        # Check early stoppage
        if p.stopping_criterion(model, val_metrics[p.optim_metric]):
            early_stop = True
            break
    
    # Load in best model if training finished
    if not early_stop:
        p.stopping_criterion.load_best_model(model)
    
    
    # Test set evaluation
    test_metrics = test(model, p, test_loader)
    print("Training finished")
    pprint(test_metrics)

    return


def test(model, p, data_loader, threshold=None):
    _threshold = p.threshold if (threshold is None) else threshold
    confusion_matrix = torch.zeros((2,2))
    model.eval()
    with torch.no_grad():
        for img, labels in data_loader:

            img = img.to(p.device, dtype=torch.float32) / 255.0 # [N,C,H,W] float32
            labels = labels.to(p.device) # [N,1,H,W] uint8

            output_logits = model(img) # [N,1,H,W]
            probability_map = F.sigmoid(output_logits)
            binary_map = 255*(probability_map >= _threshold).type(torch.uint8)

            cm = metrics.confusion_matrix(binary_map, labels)
            confusion_matrix += cm

    return metrics.calculate_metrics(confusion_matrix)


def tune_threshold(N, lower, upper, model, p, data_loader):

    thresholds = np.linspace(lower,upper,N)
    scores = np.zeros(N)

    for idx,theta in enumerate(thresholds):
        test_score = test(model, p, data_loader, threshold=theta)
        scores[idx] = test_score[p.optim_metric].item()
    
    print(f"Scores {scores}")
    
    return thresholds[np.argmax(scores)]