import os
import torch.optim
from tqdm import tqdm
import matplotlib.pyplot as plt


@torch.no_grad()
def get_accuracy(model, X, y, batch_size):
    n = X.shape[0]
    n_batches = int(n / batch_size)
    correct = 0
    total = 0
    for batch in range(n_batches):
        batch_start, batch_end = batch * batch_size, (batch + 1) * batch_size
        X_batch, y_batch = X[batch_start : batch_end], y[batch_start : batch_end]
        output = model(X_batch)
        y_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += y_pred.eq(y_batch.view_as(y_pred)).sum().item()
    return correct / n


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }, path)


def save_training_curve_plot(iter_l, epoch_l, loss_l, acc_train_l, acc_val_l, path):
    # loss
    plt.title("Training Curve")
    plt.plot(iter_l, loss_l, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(path, 'training_loss.png'))
    plt.clf();
    # accuracy
    plt.title("Training Curve")
    plt.plot(epoch_l, acc_train_l, label="Train")
    plt.plot(epoch_l, acc_val_l, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.savefig(os.path.join(path, 'training_accuracy.png'))
    plt.clf();

    
def train(model, X_train, y_train, X_val, y_val,
          batch_size=64, num_epochs=1, optimizer=None,
          save_checkpoints=True, regularity_checkpoints=100, path_checkpoints="",
          resume_training=False, last_epoch=0):
    
    criterion = torch.nn.CrossEntropyLoss()
    if not optimizer:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    n_train = X_train.shape[0]
    n_batches = int(n_train / batch_size)
    
    iter_l, epoch_l, loss_l, acc_train_l, acc_val_l = [], [], [], [], []
    acc_val_best = 0

    i = 0 # the number of iterations
    if not resume_training: # if training from zero
        last_epoch = 0
    for epoch in range(last_epoch, last_epoch + num_epochs):
        print(f"EPOCH {epoch} / {num_epochs - 1}")
        epoch_l.append(epoch)
        for batch in tqdm(range(n_batches), desc="Batches: "):
            batch_start, batch_end = batch * batch_size, (batch + 1) * batch_size
            X_batch, y_batch = X_train[batch_start : batch_end], y_train[batch_start : batch_end]
            
            # updating weights
            model.train()
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            # saving training loss
            i += 1
            iter_l.append(i)
            loss_l.append(float(loss) / batch_size)

        model.eval()
        acc_train = get_accuracy(model, X_train, y_train, batch_size)
        acc_train_l.append(acc_train)
        acc_val = get_accuracy(model, X_val, y_val, batch_size)
        acc_val_l.append(acc_val)
        acc_val_best = max(acc_val_best, acc_val)
        print(f"TRAIN       accuracy: {acc_train}")
        print(f"VALIDATION  accuracy: {acc_val}")
        
        # saving checkpoints
        if save_checkpoints:
            if (epoch % regularity_checkpoints == 0) or (epoch == num_epochs - 1):
                if not os.path.exists(path_checkpoints):
                    os.makedirs(path_checkpoints)
                last_model_path = os.path.join(path_checkpoints, f'Last model.pt')
                save_checkpoint(model, optimizer, epoch, float(loss), path=last_model_path)
                save_training_curve_plot(iter_l, epoch_l, loss_l, acc_train_l, acc_val_l, path_checkpoints)
                print(f"Weights saved to {last_model_path}.")
                if acc_val == acc_val_best: # maintain the best model saved parameters
                    best_model_path = os.path.join(path_checkpoints, f'Best model.pt')
                    save_checkpoint(model, optimizer, epoch, float(loss), path=best_model_path)
                    print(f"Weights saved to {best_model_path}.")
                    
        print("\n")

    print(f"Final TRAIN       accuracy: {acc_train_l[-1]}")
    print(f"Final VALIDATION  accuracy: {acc_val_l[-1]}")
    
    return iter_l, epoch_l, loss_l, acc_train_l, acc_val_l;
