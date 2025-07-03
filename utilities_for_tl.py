import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from torch.utils.data import Dataset
import json

class Model_dynamic(nn.Module):
    def __init__(self, h1, num_layers, freeze_layers=[3]):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()  # List to hold layers

        # First layer
        self.layers.append(nn.Linear(2, h1))

        # Add hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(h1, h1))  # Each hidden layer has h1 neurons

        # Output layer
        self.out = nn.Linear(h1, 10)

        # Freeze selected layers
        for layer_idx in freeze_layers:
            if layer_idx < len(self.layers):  # Ensure valid layer index
                for param in self.layers[layer_idx].parameters():
                    param.requires_grad = False
        # # #freeze the output layer
        # for param in self.out.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        for i in range(self.num_layers):
            x = F.elu(self.layers[i](x))  # Apply ELU after each layer
        x = self.out(x)  # Output layer
        return x

def train_regression_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience, scheduler):
    # Move model to the specified device
    model.to(device)
    # To track the history of training losses
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    early_counter = 0  # early-stopping counter
    best_model_state = None
    # mae_scores = []
    mse_values = []


    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_trainable_params}")

    for epoch in range(num_epochs):
        # uncomment these lines when doing gradually unfreezing
        # if epoch == 100:
        #     #unfreeze the frozen layer
        #     # for param in model.layers[1].parameters():
        #     #     param.requires_grad = True
        #     for param in model.layers[2].parameters():
        #         param.requires_grad = True
        #     print(f"layer unfrozen at epoch {epoch}")
        #
        #     # Reduce learning rate when unfreezing
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.5  # Adjust as needed
        #     print(f"Learning rate adjusted to {optimizer.param_groups[0]['lr']:.6e}")

        model.train()  # Make model into training mode
        train_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            assert outputs.shape == targets.shape, f"Output shape {outputs.shape} doesn't match target shape {targets.shape}"
            loss = criterion(outputs, targets)

            train_loss += loss.item() * inputs.size(0) # x batch_size to account for the loss that is the avg per batch

            loss.backward()
            optimizer.step()

        # Calculate and store average loss
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation step
        model.eval() #turn into evaluation mode
        val_loss = 0
        r2_values = []
        mean_r2_scores = []
        all_predictions = []
        all_targets = []

        with torch.no_grad(): #disable gradient calc
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputsv = model(inputs)        #v from validation
                lossv = criterion(outputsv, targets)
                val_loss += lossv.item() * inputs.size(0)
                # Store predictions and targets for export
                all_predictions.append(outputsv.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

                #r^2 calculation
                for col in range(outputsv.shape[1]):
                    pred_col = outputsv[:, col].cpu().numpy()  # Get predictions for the current column
                    target_col = targets[:, col].cpu().numpy()  # Get targets for the current column
                    r2 = r2_score(target_col, pred_col)
                    r2_values.append(r2)
                    # Calculate MSE for the current column
                    mse = mean_squared_error(target_col, pred_col)
                    mse_values.append(mse)
        # ---------
        # Flatten the lists and save them to CSV
        all_predictions = np.vstack(all_predictions)  # Convert list of batches to full array
        all_targets = np.vstack(all_targets)  # Convert list of batches to full array

        # Create a DataFrame with meaningful column names
        num_targets = all_targets.shape[1]
        pred_columns = [f'Pred_{i + 1}' for i in range(num_targets)]
        target_columns = [f'Target_{i + 1}' for i in range(num_targets)]
        df_export = pd.DataFrame(np.hstack((all_predictions, all_targets)), columns=pred_columns + target_columns)

        # Save CSV file
        df_export.to_csv('validation_results.csv', index=False,sep=';')
        # ---------



        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        mean_r2 = np.mean(r2_values)
        mean_mse = np.mean(mse_values)  # Calculate mean MSE for all columns
        mean_r2_scores.append(mean_r2)
        # Adjust learning rate with ReduceLROnPlateau
        if scheduler:
            scheduler.step(val_loss)
            current_lr = scheduler.get_last_lr()[0]  # Get the learning rate of the first group
        else:
            current_lr = optimizer.param_groups[0]['lr']  # Fallback if no scheduler is provided

        if epoch % 10 == 0 or epoch == num_epochs-1: # print metrics
            print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Learning Rate: {current_lr:.4e}" )
            for col in range(outputs.shape[1]):
                print(f"  R^2 for column {col + 1}: {r2_values[col]:.4f}")
            print(f"  Mean R²: {mean_r2:.4f}")
            for col in range(outputs.shape[1]):
                print(f"  MSE for column {col + 1}: {mse_values[col]:.4f}")
            print(f"  Mean MSE for all columns: {mean_mse:.4f}")

        # Early stopping implementation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_counter = 0  # reset the counter if validation loss improves
            best_model_state = model.state_dict()
        else:
            early_counter += 1
            if early_counter >= patience:
                print(f"Early stopping triggered at {epoch + 1} epoch")
                model.load_state_dict(best_model_state)
                break

    return model, train_losses, val_losses, mean_r2