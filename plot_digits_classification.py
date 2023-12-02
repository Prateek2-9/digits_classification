# utils import
from utils import load_dataset, data_preprocessing, split_train_dev_test, predict_and_eval
from utils import get_list_of_param_comination, tune_hparams
import pandas as pd
from joblib import load, dump
import argparse
from sklearn.model_selection import cross_val_score
from statistics import mean, stdev

parser = argparse.ArgumentParser()
parser.add_argument("--total_run", type=int)
parser.add_argument("--dev_size", type=float)
parser.add_argument("--test_size", type=float)
parser.add_argument("--prod_model_path", type=str, default=None)
parser.add_argument("--model_type", type=str)

args = parser.parse_args()

# 1. get/load the dataset
X, y = load_dataset()

# 2. Sanity check of data

# Taking different combinations of train dev and test and reporting results

results = []
for run_num in range(args.total_run):
    for ts in [args.test_size]:
        for ds in [args.dev_size]:
            # 3. Splitting the data
            X_train, y_train, X_test, y_test, X_dev, y_dev = split_train_dev_test(X, y, test_size=ts, dev_size=ds)

            # 4. Preprocessing the data
            X_train = data_preprocessing(X_train)
            X_test = data_preprocessing(X_test)
            X_dev = data_preprocessing(X_dev)

            # 5. Classification model training
            # 5.1 SVM
            # Hyperparameter tuning for gamma and C
            if args.model_type == 'svm':
                gamma_values = [0.0001, 0.001, 0.005, 0.01]
                C_values = [0.1, 0.5, 1]
                list_of_param_combination = get_list_of_param_comination([gamma_values, C_values], ['gamma', 'C'])
                best_hparams, best_model, best_val_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev,
                                                                           list_of_param_combination, model_type=args.model_type)

                # Get training accuracy of this best model:
                train_accuracy, candidate_pred = predict_and_eval(best_model, X_train, y_train)

                # 6. Prediction and evaluation on test set
                # Test accuracy
                test_accuracy, candidate_pred = predict_and_eval(best_model, X_test, y_test)

                # Print for debugging
                print(f'svm model  test_size={ts} dev_size={ds} train_size={round(1 - ts - ds, 2)} '
                      f'train_acc={train_accuracy} dev_acc={best_val_accuracy} test_acc={test_accuracy} '
                      f'best_hyper_params={best_hparams}')

                # Explicitly assign 'model_type'
                current_model_type = args.model_type

                results.append({'run_num': run_num, 'model_type': current_model_type, 'train_accuracy': train_accuracy,
                                'val_accuracy': best_val_accuracy, 'test_acc': test_accuracy,
                                'best_hparams': best_hparams})
            # 5.2 Decision Tree
            # Hyperparameter tuning
            elif args.model_type == 'tree':
                max_depth = [5, 10, 20, 50]
                list_of_param_combination = get_list_of_param_comination([max_depth], ['max_depth'])
                best_hparams, best_model, best_val_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev,
                                                                           list_of_param_combination,
                                                                           model_type=args.model_type)

                # Get training accuracy of this best model:
                train_accuracy, candidate_pred = predict_and_eval(best_model, X_train, y_train)

                # 6. Prediction and evaluation on test set
                # Test accuracy
                test_accuracy, candidate_pred = predict_and_eval(best_model, X_test, y_test)

                # Print for debugging
                print(f'tree model test_size={ts} dev_size={ds} train_size={round(1 - ts - ds, 2)} '
                      f'test_acc={test_accuracy} best_hyper_params={best_hparams}')

                # Explicitly assign 'model_type'
                current_model_type = args.model_type

                results.append({'run_num': run_num, 'model_type': current_model_type, 'train_accuracy': train_accuracy,
                                'val_accuracy': best_val_accuracy, 'test_acc': test_accuracy,
                                'best_hparams': best_hparams})

            # 5.3 Logistic Regression
            elif args.model_type == 'lr':
                solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                for solver in solvers:
                    # Hyperparameter tuning for Logistic Regression
                    list_of_param_combination = get_list_of_param_comination([[solver]], ['solver'])
                    best_hparams, best_model, best_val_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, list_of_param_combination, model_type=args.model_type)

                    # Get training accuracy of this best model
                    train_accuracy, candidate_pred = predict_and_eval(best_model, X_train, y_train)

                    ################################################################################################
                    # 6. Prediction and evaluation on the test set
                    # Test accuracy
                    test_accuracy, candidate_pred = predict_and_eval(best_model, X_test, y_test)

                    # Print for GitHub actions
                    print(f'lr model, solver={solver}, test_size={ts}, dev_size={ds}, train_size={round(1 - ts - ds, 2)}, train_acc={train_accuracy}, dev_acc={best_val_accuracy}, test_acc={test_accuracy}, best_hyper_params={best_hparams}')

                    # Save the model
                    roll_no = "m22aie214"  # Replace with your actual roll number
                    model_path = f"models/{roll_no}_lr_{solver}.joblib"
                    dump(best_model, model_path)

                    # Bonus: Cross-validation
                    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
                    print(f'Mean CV Accuracy for solver={solver}: {mean(cv_scores)}, Std CV Accuracy: {stdev(cv_scores)}')

                    results.append({'run_num': run_num, 'model_type': args.model_type, 'solver': solver, 'train_accuracy': train_accuracy, 'val_accuracy': best_val_accuracy, 'test_acc': test_accuracy, 'best_hparams': best_hparams})


            if args.prod_model_path is not None:
                prod_model = load(args.prod_model_path)
                prod_test_accuracy, prod_model_pred = predict_and_eval(prod_model, X_test, y_test)
                print(f'prod model test_size={ts} dev_size={ds} train_size={round(1 - ts - ds, 2)} '
                      f'test_acc={test_accuracy}')

                # Explicitly assign 'model_type'
                current_model_type = args.model_type

                results.append({'run_num': run_num, 'model_type': current_model_type, 'train_accuracy': train_accuracy,
                                'val_accuracy': best_val_accuracy, 'test_acc': test_accuracy,
                                'best_hparams': best_hparams})

# Convert the list of results to a DataFrame
results_df = pd.DataFrame(results)

# Print the DataFrame to check its structure
print("Results DataFrame:")
print(results_df)

# Check if 'model_type' is present in the DataFrame
if 'model_type' in results_df.columns:
    # Group by 'model_type' and print summary statistics
    print("\nSummary Statistics:")
    print(results_df.groupby('model_type').describe().T)
else:
    print("Error: 'model_type' column not found in DataFrame.")
