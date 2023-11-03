"""
Author: Prateek 
digit classification model code for mlops 

"""

#utils import
from utils import load_dataset, data_preprocessing, split_train_dev_test,predict_and_eval
from utils import train_model
from utils import get_list_of_param_comination, tune_hparams
import pandas as pd 
import sys
import joblib
from joblib import dump, load
from sklearn.metrics import confusion_matrix, classification_report,f1_score

# if __name__ == "__main__":
#     if len(sys.argv) != 6:
#         print("Usage: python plot_digits_classification.py <total_run> <dev_size> <test_size> <prod_model_path> <model_type>")
#         sys.exit(1)

#     total_run = int(sys.argv[1])
#     dev_size = float(sys.argv[2])
#     test_size = float(sys.argv[3])
#     prod_model_path = sys.argv[4]
#     model_type = sys.argv[5]


total_run = int(sys.argv[1])
dev_size = [float(sys.argv[2])]
test_size = [float(sys.argv[3])]
prod_model_path = sys.argv[4]
model_type = sys.argv[5]


###########################################################################################
#1.get/load the dataset
X,y = load_dataset()
###for quiz1 
# print("total no of images in datasat",y.shape[0])
# print("size of each image in datasat ", X[0].shape)

#2.Sanity check of data

################################################################################################
#taking different combinations of train dev and test and reporting results

# test_size =  [0.2]
# dev_size = [0.2]
# total_run = 5
results = []
for run_num in range(total_run):
    for ts in test_size:
        for ds in dev_size:
            #3. Spliting the data
            X_train, y_train, X_test, y_test, X_dev, y_dev = split_train_dev_test(X, y, test_size=ts, dev_size=ds)  

            #################################################################################################
            #4. Preprocessing the data
            X_train = data_preprocessing(X_train)
            X_test= data_preprocessing(X_test)
            X_dev =data_preprocessing(X_dev)

            #################################################################################################
            #5. Classification model training
            #5.1 SVM - hyperparameter tuning for gamma and C
            if model_type == 'svm':
                gamma_values = [0.001, 0.002, 0.005, 0.01, 0.02]
                C_values = [0.1, 0.2, 0.5, 0.75, 1]
                list_of_param_comination = get_list_of_param_comination([gamma_values, C_values],  ['gamma', 'C'])

                for svm_gamma in gamma_values:
                    for C_value in C_values:
                        # Train a model with the current hyperparameters
                        model_params = {'gamma': svm_gamma, 'C': C_value}
                        model_filename = f"svm_gamma_{svm_gamma}_C_{C_value}.joblib"
                        model = train_model(X_train, y_train, {'gamma': svm_gamma, 'C': C_value}, model_type, model_filename)

                        # Save the model with a unique filename based on the hyperparameters
                        model_filename = f"svm_gamma_{svm_gamma}_C_{C_value}.joblib"
                        joblib.dump(model, f'/digits/models/{model_filename}')

                        # Get training accuracy of this model
                        train_accuracy, candidate_pred = predict_and_eval(model, X_train, y_train)

                        # Evaluation on the test set
                        test_accuracy, candidate_pred = predict_and_eval(model, X_test, y_test)

                        # Print for GitHub Actions
                        print('candidate svm model  ', 'test_size=', ts, ' dev_size=', ds, ' train_size=', round(1 - ts - ds, 2),
                            f' svm_gamma={svm_gamma}, C={C_value}, train_acc={train_accuracy}, test_acc={test_accuracy}')
                        results.append({'model_type': model_type, 'train_accuracy': train_accuracy, 'test_acc': test_accuracy, 'svm_gamma': svm_gamma, 'C': C_value})


                # Initialize variables to store the best model and hyperparameters
                best_hparams = None
                best_model = None
                best_val_accuracy = -1

                for hparams in list_of_param_comination:
                    model = train_model(X_train, y_train, hparams, model_type, model_filename)
                    val_accuracy, _ = predict_and_eval(model, X_dev, y_dev)
                    
                    if val_accuracy > best_val_accuracy:
                        best_hparams = hparams
                        best_model = model
                        best_val_accuracy = val_accuracy

                # Save the best model with the desired filename
                svm_gamma = best_hparams['gamma']
                C_value = best_hparams['C']
                model_filename = f"svm_gamma: {svm_gamma} C:{C_value}.joblib"
                joblib.dump(best_model, f'/digits/models/{model_filename}')

                # Get training accuracy of this best model
                train_accuracy, candidate_pred = predict_and_eval(best_model, X_train, y_train)

                # Evaluation on the test set
                test_accuracy, candidate_pred = predict_and_eval(best_model, X_test, y_test)

                # Print for GitHub Actions
                print('candidate svm model  ', 'test_size=', ts, ' dev_size=', ds, ' train_size=', round(1 - ts - ds, 2),
                    ' train_acc=', train_accuracy, ' dev_acc', best_val_accuracy, ' test_acc=', test_accuracy, ' best_hyper_params=', best_hparams)
                results.append({'run_num': run_num, 'model_type': model_type, 'train_accuracy': train_accuracy, 'val_accuracy': best_val_accuracy,
                                'test_acc': test_accuracy, 'best_hparams': best_hparams})

            # ... (the rest of your script)

            #5.2 Decision Tree
            #hyper parameter tunning
            if model_type == 'tree':
                max_depth = [5,10,20,30,50,75,100]
                list_of_param_comination = get_list_of_param_comination([max_depth],  ['max_depth'])
                best_hparams, best_model, best_val_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, list_of_param_comination,model_type = model_type)

                #get training accuracy of this best model:
                train_accuracy, candidate_pred = predict_and_eval(best_model, X_train, y_train)

                ################################################################################################
                #6. Prediction and evaluation on test sat
                # test accuracy
                test_accuracy, candidate_pred= predict_and_eval(best_model, X_test, y_test)

                #print for github actions
                print('candidate tree model ','test_size=',ts,' dev_size=',ds,' train_size=',round(1-ts-ds,2),' test_acc=',test_accuracy, ' best_hyper_params=', best_hparams)
                results.append({'run_num':run_num,'model_type':model_type, 'train_accuracy':train_accuracy, 'val_accuracy':best_val_accuracy,
                                'test_acc':test_accuracy, 'best_hparams':best_hparams})
            if prod_model_path is not None:
                prod_model = load(prod_model_path)
                prod_test_accuracy, prodmodel_pred = predict_and_eval(prod_model, X_test, y_test)
                print('prod model ','test_size=',ts,' dev_size=',ds,' train_size=',round(1-ts-ds,2),', test_acc=',test_accuracy)

            cm = confusion_matrix(prodmodel_pred, candidate_pred)
            print("confuson matrix btw prod and candidate model ", cm)

            f1_scores = f1_score(y_test, candidate_pred, average=None)
            # Calculate the macro-average F1 score
            macro_avg_f1 = sum(f1_scores) / len(f1_scores)
            print('macro avrege f1 score for candidate model : ', macro_avg_f1)

            f1_scores = f1_score(y_test, prodmodel_pred, average=None)
            # Calculate the macro-average F1 score
            macro_avg_f1 = sum(f1_scores) / len(f1_scores)
            print('macro avrege f1 score for prod model : ', macro_avg_f1)


# result_df = pd.DataFrame(results)
# print(result_df.groupby('model_type').describe().T)