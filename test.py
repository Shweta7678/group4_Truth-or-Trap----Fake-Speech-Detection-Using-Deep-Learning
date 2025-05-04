import argparse
import sys
import os
import warnings
import torch
from torch.utils.data import DataLoader
from model import Model
from data_utils_SSL import genSpoof_list,Dataset_ASVspoof2021_eval
from core_scripts.startup_config import set_random_seed
from tqdm import tqdm 
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=10, num_workers = 8, shuffle=False, drop_last=False)
    
    print(f"Number of files in dataset: {len(dataset)}")  # Add this to check the dataset size
    model.eval()
    
    # Initialize lists to store predictions and true labels
    all_test_preds = []
    all_test_labels = []
    with torch.no_grad():
        for batch_x, labels in tqdm(data_loader):
            # print(f"Processing batch with {len(utt_id)} files")  # Add this to check batches
            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)
            labels = labels.to(device)
            
            batch_out = model(batch_x)
            _, predicted_class = torch.max(batch_out, 1)
            
            all_test_preds.extend(predicted_class.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())

    # # Convert lists to NumPy arrays
    # all_test_preds = np.array(all_test_preds)
    # all_test_labels = np.array(all_test_labels)

    # # Compute accuracy
    # accuracy = (all_test_preds == all_test_labels).sum() / len(all_test_labels)
    # print(f"Accuracy: {accuracy:.4f}")

    # num_classes = 2
    # conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # for true_label, pred_label in zip(all_test_labels, all_test_preds):
    #     conf_matrix[true_label, pred_label] += 1

    # print("Confusion Matrix:")
    # print(conf_matrix)




    # Define the class names based on the label encoding used in the dataset (0: fake, 1: real)
    target_names = ['fake', 'real']

    # Generate and print the classification report
    print("Classification Report on Test Data:")
    report = classification_report(all_test_labels, all_test_preds, target_names=target_names)
    print(report)

    # Generate and print the confusion matrix
    print("\nConfusion Matrix on Test Data:")
    cm = confusion_matrix(all_test_labels, all_test_preds)
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()



def main():
    parser = argparse.ArgumentParser(description='ASVspoof2021 testing system')
    # Dataset paths and other args
    parser.add_argument('--database_path', type=str, default='/home/pm_students/SSL_Anti-spoofing/archive/for-norm/for-norm/ASVspoof2021_LA_eval/wav')
    parser.add_argument('--protocols_path', type=str, default='database/')
    parser.add_argument('--model_path', type=str, default='/home/pm_students/SSL_Anti-spoofing/models/model_LA_WCE_100_8_1e-06/epoch_4', help='Path to the trained model')
    parser.add_argument('--eval_output', type=str, default='evaluation_results.txt', help='Path to save the evaluation result')
    parser.add_argument('--track', type=str, default='LA', choices=['LA', 'PA', 'DF'], help='Track name (LA/PA/DF)')
    parser.add_argument('--eval', action='store_true', default=False, help='Whether to perform evaluation')

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    
    # Set the seed
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    # Initialize model
    model = Model(args, device)
    model = model.to(device)
    
    # Load model checkpoint if provided
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f'Model loaded: {args.model_path}')
    
    # Prepare evaluation dataset
    eval_file = "/home/pm_students/SSL_Anti-spoofing/archive/for-norm/for-norm/ASVspoof2021_LA_eval/ASVspoof2021.LA.cm.eval.trl.txt"
    label_eval, file_eval = genSpoof_list( dir_meta =  eval_file,is_train=False,is_eval=False)
    print('no. of eval trials',len(file_eval))
    eval_set = Dataset_ASVspoof2021_eval(file_eval, base_dir="/home/pm_students/SSL_Anti-spoofing/archive/for-norm/for-norm/ASVspoof2021_LA_eval/wav",labels=label_eval)
    print(f"test_set view:> {eval_set[0]}")
    # Perform evaluation and save results
    produce_evaluation_file(eval_set, model, device, args.eval_output)
    print('Evaluation completed.')


if __name__ == '__main__':
    main()

