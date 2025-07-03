import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset_class_V3_minmax import MergedDatasetTest, unscale_min_max
import json
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utilities import setup_device, test_model, Model, unscale, calculate_weighted_mse, ScaledDataset, calculate_r2
from utilities_for_tl import Model_dynamic
import os

#specify test data, neural network and folder, change them with every test
current_directory = os.path.dirname(os.path.realpath(__file__))
csv_file_path_test = os.path.join(current_directory,'trained_model_Ar_weighted_tuned_JAN25(v02_physics_based)', 'test_data_no_head_outer_corner_Ar.csv')
csv_file_path_train = os.path.join(current_directory,'trained_model_Ar_weighted_tuned_JAN25(v02_physics_based)', 'train_data_no_head_outer_corner_Ar.csv')
# csv_file_path_test = os.path.join(current_directory, 'split_data_tests', 'argon02', 'test_Ar_subset_80.csv' )
#save paths for graphs that needed for transfer learning
folder = 'M12_freeze3'
save_path1 = os.path.join(current_directory, 'transfer_learning_graphs3_layers_O2', folder, 'R2.png')
save_path2 = os.path.join(current_directory, 'transfer_learning_graphs3_layers_O2', folder, 'residuals.png')
neural_network = 'M2_trained_model2_Ar_with_Ar_data_minmax.pth'

h1_val = 10
layers = 3
batch = 128
xlim = 100
statsjson = 'overall_min_max.json'#os.path.join(current_directory,'trained_model_Ar_weighted_tuned_JAN25(v02_physics_based)','column_statsAr.json') #also change the stats file at the MergedDatasetTest
name_of_predictions = '1predictions.csv' #csv file for unscaled predictions to be saved
#make the calculations at the test set
device = setup_device()
basic_model = Model_dynamic(h1=h1_val, num_layers=layers, freeze_layers=[])
dataset_test = MergedDatasetTest(csv_file_path_test,statsjson) #!!don't forget to change the stats file at the MergedDatasetTest
test_loader = DataLoader(dataset_test, batch_size=batch, shuffle=False)
basic_model.load_state_dict(torch.load(neural_network)) #trained_model1_O2_weighted_mse.pth change O2 to Ar if you want to choose the other dataset
trained_model = basic_model
criterion = calculate_weighted_mse(reduction='mean') #nn.SmoothL1Loss() #calculate_huber_loss #
# Evaluate on the test set
test_loss, all_predictions, all_targets, r22, elapsed_time = test_model(model=trained_model,
                                                     test_loader=test_loader,
                                                     criterion=criterion,
                                                     device=device)
print(f"mean test loss: {test_loss:.4f}")
print(f"Time taken for evaluation: {elapsed_time:.5f} seconds")
with open(statsjson, 'r') as f:
    scaling_info = json.load(f)
output_columns_names = list(scaling_info.keys())
output_columns_names = [col for col in output_columns_names if col not in ['Power', 'Pressure']]
#perform unscale of the predictions
unscaled_predictions = unscale_min_max(all_predictions, output_columns_names, scaling_info)
# unscaled_predictions = unscale(all_predictions, output_columns_names, scaling_info)
predictions_df = pd.DataFrame(unscaled_predictions, columns=output_columns_names)
predictions_df.to_csv(name_of_predictions, index=False, sep=';')

#now both csv's of test data and predictions should be read for r^2 score
model_predictions = pd.read_csv(name_of_predictions, sep=';')
real_targets = pd.read_csv(csv_file_path_test,
                           usecols=lambda column: column not in ['Power', 'Pressure'], sep=';')

#manual r^2 calculation
r2_scores = []
r2_scores_sklearn = []
for col in range(model_predictions.shape[1]):  # Loop through columns
    pred_col = model_predictions.iloc[:, col]  # Get predictions for the current column
    target_col = real_targets.iloc[:, col]  # Get targets for the current column
    # Compute R^2 for the current column pair
    r2 = calculate_r2(target_col, pred_col)
    r2_scores.append(r2)
    # Compute R^2 using sklearn
    r2_sklearn = r2_score(target_col, pred_col)
    r2_scores_sklearn.append(r2_sklearn)

plt.figure(figsize=(10.5, 7.5))  # Set the size of the figure
plt.bar(list(range(1, len(r2_scores) + 1)), r2_scores, color='skyblue', edgecolor='black')
# Add labels and title
plt.xlabel('Etching rate point', fontsize=20)
plt.ylabel(r"$R^2$ Score", fontsize=20)
# plt.title('R^2 Scores for Each Column Pair (Prediction vs Target)', fontsize=22)
plt.xticks(list(range(1, len(r2_scores) + 1)))  # Set x-ticks for each column pair
plt.tight_layout()
plt.tick_params(axis='both', which='major', labelsize=20)
######## plt.savefig(save_path1, dpi=600, bbox_inches='tight')
plt.show()

# Visualize distribution of residuals
residuals = abs(real_targets - model_predictions)/real_targets * 100
residuals_flatten = residuals.values.flatten()
plt.rcParams.update({'font.size': 20})  # Adjust font size as needed
plt.figure(figsize=(10.5, 7.5))
# sns.histplot(residuals, kde=True, color='blue')  # KDE to show the distribution
# KDE plot
ax = sns.kdeplot(residuals, color='blue', fill=False)

# Remove "O" from legend labels
new_labels = [label.get_text().replace("O", "") for label in ax.get_legend().get_texts()]
ax.legend(new_labels)
# plt.title('Histogram of Residuals')
plt.xlabel('(physics_based – NN)/physics_based %', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
# Set axis limits
# plt.ylim(0, 0.10)  # Set the y-axis maximum to 0.10
plt.xlim(-1, xlim)    # Set the x-axis to extend to 15%
plt.rcParams.update({'font.size': 18})
plt.tick_params(axis='both', which='major', labelsize=20)
####### plt.savefig(save_path2, dpi=600, bbox_inches='tight')
plt.show()

residuals_df = pd.DataFrame({'Residuals (%)': residuals_flatten})

from sklearn.linear_model import LinearRegression

# Define input and output columns
input_columns = ['Power', 'Pressure']
output_columns = output_columns_names  # 10 targets

# Load training data for baseline model
train_df = pd.read_csv(csv_file_path_train, sep=';')
X_train = train_df[input_columns]
y_train = train_df[output_columns]

# Train linear regression on training set
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Load test inputs
X_test = pd.read_csv(csv_file_path_test, usecols=input_columns, sep=';')
y_test = real_targets.copy()
# Compute R^2 for each output
y_pred_lr = lin_reg.predict(X_test)
r2_lr = [r2_score(y_test.iloc[:, i], y_pred_lr[:, i]) for i in range(y_test.shape[1])]


# Plot comparison of NN vs Linear Regression R²
x_labels = list(range(1, len(r2_lr) + 1))
bar_width = 0.35
plt.figure(figsize=(11, 7))
plt.bar([x - bar_width/2 for x in x_labels], r2_scores_sklearn, width=bar_width, label='Neural Network', color='skyblue')
plt.bar([x + bar_width/2 for x in x_labels], r2_lr, width=bar_width, label='Linear Regression', color='salmon')  #'salmon' 'rosybrown'
plt.xlabel('Etching rate point', fontsize=20)
plt.ylabel(r"$R^2$ Score", fontsize=20)
plt.xticks(x_labels)
plt.legend(fontsize=18)
plt.tight_layout()
plt.tick_params(axis='both', which='major', labelsize=20)
plt.savefig('nn_vs_lr_r2_comparison.png', dpi=600, bbox_inches='tight')
plt.show()

mean_r2_nn = np.mean(r2_scores_sklearn)
mean_r2_lr = np.mean(r2_lr)

print(f"\nAverage R² score of Neural Network: {mean_r2_nn:.4f}")
print(f"Average R² score of Linear Regression: {mean_r2_lr:.4f}")