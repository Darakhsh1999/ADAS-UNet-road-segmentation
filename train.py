import torch
from tqdm import tqdm


def train(model, optimizer, p, train_loader, val_loader, test_loader):

    ### Train model 
    for epoch_idx in tqdm(range(p.n_epochs)):

        # Loop through training data
        epoch_loss = 0.0
        model.train()
        for img, labels in train_loader:
            
            img = img.to(p.device) 
            labels = labels.to(p.device)

            optimizer.zero_grad()

            output = model(img) 
            loss = p.loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch {1+epoch_idx} loss = {epoch_loss:.4f}")

        # Validation
        model.eval()
        val_metrics = test(model, p, val_loader)


def test(model, p, data_loader):
    model.eval()
    n_correct_predictions = 0.0
    with torch.no_grad():
        for img, labels in data_loader:

            img, labels = img.to(p.device), labels.to(p.device)
            output_probability = model(img) # (N,10)

            predicted_batch_class = torch.argmax(output_probability, dim=-1) # (N,) class 0-9

            n_correct_predictions += (predicted_batch_class == labels).sum().cpu().item()

    accuracy = n_correct_predictions / len(test_data)
    print(f"Test accuracy = {accuracy*100:.2f}%")
    pass