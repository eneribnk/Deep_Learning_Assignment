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
import time
# Device setup (CUDA or CPU)
def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device is {device}.')
    return device


# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_regression_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience, scheduler):
    # Move model to the specified device
    model.to(device)
    r2_first_5 = [] #vector to track the r2 for the first 5 columns
    # To track the history of training losses
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    early_counter = 0  # early-stopping counter
    best_model_state = None
    # mae_scores = []
    mse_values = []
    best_r2 = -float('inf')
    for epoch in range(num_epochs):
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


def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    r2_values = []
    mse_values = []
    all_predictions, all_targets = [], []
    with torch.no_grad():
        start_time = time.time()
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputst = model(inputs)
            losst = criterion(outputst, targets)
            test_loss += losst.item() * inputs.size(0)
            all_predictions.append(outputst.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            #r^2, mse calculation
            for col in range(outputst.shape[1]):
                pred_col = outputst[:, col].cpu().numpy()
                target_col = targets[:, col].cpu().numpy()  # Get targets for the current column
                r2 = r2_score(target_col, pred_col)
                r2_values.append(r2)
                mse = mean_squared_error(target_col, pred_col)
                mse_values.append(mse)
                # print(f"  R^2 for column {col + 1}: {r2:.4f}")
                # print(f"  MSE for column {col + 1}: {mse:.4f}")


            mean_r2 = np.mean(r2_values)
            mean_mse = np.mean(mse_values)
        end_time = time.time()

    elapsed_time = end_time - start_time
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    # Exclude the 9th and 10th output columns (index 8 and 9)
    filtered_predictions = all_predictions[:, :8]  # Keep only first 8 columns
    filtered_targets = all_targets[:, :8]  # Keep only first 8 columns

    # Compute overall MSE, MAE, and R² for the first 8 outputs
    overall_mse = mean_squared_error(filtered_targets, filtered_predictions)
    overall_mae = mean_absolute_error(filtered_targets, filtered_predictions)
    overall_r2 = r2_score(filtered_targets, filtered_predictions)

    print(f"\nOverall Performance Metrics:")
    print(f"  Mean MSE: {overall_mse:.7f}")
    print(f"  Mean MAE: {overall_mae:.7f}")
    print(f"  Mean R²: {overall_r2:.5f}")
    test_loss /= len(test_loader.dataset)
    return test_loss, all_predictions, all_targets, r2_values, elapsed_time


class Model(nn.Module):
    # input layer ( 2 features, Power Pressure) --> Hidden Layer 1 (H1) --> H2 --> Output (25 outputs)
    def __init__(self, in_features=2, h1=10, h2=10, out_features=10):
        super().__init__()  # instantiate our nn.Module, always have to do it
        self.fc1 = nn.Linear(in_features, h1) # we suppose fully connected layer (fc-> fully connected)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    # we need now the function to move everything forward
    def forward(self, x):
        # we choose relu
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.out(x)
        return x


def unscale(data, column_names, scaling_info):
    unscaled_data = data
    num_columns = data.shape[1]
    for i, column in enumerate(column_names):
        if i >= num_columns:  # Check if index exceeds the number of columns
            print(f"Warning: Index {i} is out of bounds for data with {num_columns} columns.")
            break
        mean = scaling_info[column]['mean']
        std = scaling_info[column]['std']
        unscaled_data[:, i] = (data[:, i] * std) + mean  # Reverse normalization
    return unscaled_data

#manual loss functions
class calculate_weighted_mse:

    def __init__(self, reduction): #i can choose for sum of errors or mean of errors or none for elemntwise
        self.reduction = reduction

    def forward(self, input, target):
        # Step 2: Check that input and target have the same shape
        if input.shape != target.shape:
            raise ValueError("Input and target must have the same shape")
        # Step 3: Compute the squared differences (squared error)
        #weights = torch.tensor([2,0.5,2,0.1,0.8,2,0.1,2,1,0.8]) for oxygen model randomized columns
        weights = torch.tensor([2,2,2,2,1,0.8,0.5,0.5,0.1,0.1]) #for oxygen model 1-10 columns
        squared_error = (input - target) ** 2 * weights
        #weighted_squared_error = squared_error * weights

        # Step 4: Apply the reduction method (mean, sum, or none)
        if self.reduction == 'mean':
            return squared_error.mean()  # Mean of squared errors
        elif self.reduction == 'sum':
            return squared_error.sum()  # Sum of squared errors
        else:
            return squared_error  # (element-wise squared error)
    def __call__(self, input, target):
        return self.forward(input, target)

class calculate_huber_loss:
    def __init__(self, delta: float = 1.0):

        #delta (float): The threshold at which to switch between squared loss and absolute loss.

        super(calculate_huber_loss, self).__init__()
        self.delta = delta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Compute the absolute error
        error = torch.abs(y_true - y_pred)

        # Calculate Huber loss based on the threshold delta
        loss = torch.where(error <= self.delta,
                           0.5 * error ** 2,  # Squared loss
                           self.delta * (error - 0.5 * self.delta))  # Absolute loss

        return loss.mean()  # Return the mean loss over the batch
#dynamic neural network to be used at the hyperparameter search
class Model_dynamic(nn.Module):
    def __init__(self, h1, num_layers):
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

    def forward(self, x):
        for i in range(self.num_layers):
            x = F.elu(self.layers[i](x))  # Apply ELU after each layer
        x = self.out(x)  # Output layer
        return x

def calculate_r2(true_values, predicted_values):
    # Ensure true_values and predicted_values are numpy arrays
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    print(true_values)
    print(predicted_values)

    # Calculate the total sum of squares (TSS)
    mean_true = np.mean(true_values)
    total_sum_of_squares = np.sum((true_values - mean_true) ** 2)

    # Calculate the residual sum of squares (RSS)
    residual_sum_of_squares = np.sum((true_values - predicted_values) ** 2)

    # Calculate R^2
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r2
#class to import the dataset and train using min max
class ScaledDataset(Dataset):
    def __init__(self, csv_file, scaler=None, is_training=True, save_scaler_file='scaler_file.pk1', stats_file='scaler_stats.json'):
        df = pd.read_csv(csv_file, sep=';')
        #split into features and targets
        X_columns = ['Power', 'Pressure']
        y_columns = [col for col in df.columns if col not in X_columns]
        self.X = df[X_columns]
        self.y = df[y_columns]

        #Code for application of minmax scaler
        if is_training:
            self.scaler = MinMaxScaler()
            self.X = pd.DataFrame(self.scaler.fit_transform(self.X), columns=X_columns)
            import joblib
            joblib.dump(self.scaler, save_scaler_file)
            stats = {col: {"min": float(self.scaler.data_min_[i]), "max": float(self.scaler.data_max_[i])}
                     for i, col in enumerate(X_columns)}
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=4)
        else:
            from joblib import load
            self.scaler = load(save_scaler_file)
            self.X = pd.DataFrame(self.scaler.transform(self.X), columns=X_columns)
        self.df = pd.concat([self.X, self.y], axis=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
            # Return a single data point
        return torch.tensor(self.X.iloc[idx].values, dtype=torch.float32), torch.tensor(self.y.iloc[idx].values, dtype=torch.float32)