import numpy as np
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
outputdata = ""

def open_file_dialog():
    plt.close()
    output_label.configure(text="Wybranie pliku automatycznie uruchomi obliczenia. Wynik pojawi się poniżej tej linii.\nPodczas porówywania modeli znak > oznacza że model jest bardziej trafny/lepszy a nie zmienna jest większa. (analogiczne dla = i <)\n\n")
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        load_data(file_path)

def confusion_matrix(labels, predictions):
    tp = np.sum((labels == '>50K') & (predictions == '>50K'))
    tn = np.sum((labels == '<=50K') & (predictions == '<=50K'))
    fp = np.sum((labels == '>50K') & (predictions == '<=50K'))
    fn = np.sum((labels == '<=50K') & (predictions == '>50K'))
    return np.array([[tn, fp], [fn, tp]])

def roc_curve(labels, scores):
    sorted_indexes = np.argsort(scores, kind='mergesort')[::-1]
    sorted_labels = labels[sorted_indexes]
    tpr = np.cumsum(sorted_labels == 1) / np.sum(labels == 1)
    fpr = np.cumsum(sorted_labels == 0) / np.sum(labels == 0)
    return fpr, tpr

def auc(fpr, tpr):
    return np.sum(np.diff(fpr) * tpr[:-1])

def calculate_metrics(true_values, predictions):
    errors = true_values - predictions
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(errors / true_values)) * 100
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    return mae, mape, mse, rmse

def norm(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

def compare(x, y):
    if x > y:
        return '>'
    elif x < y:
        return '<'
    else:
        return '='

def load_data(file_path):
    data = pd.read_csv(file_path)
    if selected_option.get() == 1:
        true_labels = data['income']
        model1_pred_labels = data['C50_PV']
        model1_prob = data['C50_prob1'].astype(float)
        model2_pred_labels = data['rf_PV']
        model2_prob = data['rf_prob1'].astype(float)

        conf_matrix_model1 = confusion_matrix(true_labels, model1_pred_labels)
        conf_matrix_model2 = confusion_matrix(true_labels, model2_pred_labels)

        accuracy_model1 = (conf_matrix_model1[0, 0] + conf_matrix_model1[1, 1]) / np.sum(conf_matrix_model1)
        sensitivity_model1 = conf_matrix_model1[0, 0]/(conf_matrix_model1[0, 0] + conf_matrix_model1[0, 1])
        specificity_model1 = conf_matrix_model1[1, 1] / (conf_matrix_model1[1, 1] + conf_matrix_model1[1, 0])
        precision_model1 = conf_matrix_model1[0, 0] / (conf_matrix_model1[0, 0] + conf_matrix_model1[0, 1])
        f1_model1 = 2 * (precision_model1 * sensitivity_model1) / (precision_model1 + sensitivity_model1)

        accuracy_model2 = (conf_matrix_model2[0, 0] + conf_matrix_model2[1, 1]) / np.sum(conf_matrix_model2)
        sensitivity_model2 = conf_matrix_model2[0, 0] / (conf_matrix_model2[0, 0] + conf_matrix_model2[0, 1])
        specificity_model2 = conf_matrix_model2[1, 1] / (conf_matrix_model2[1, 1] + conf_matrix_model2[1, 0])
        precision_model2 = conf_matrix_model2[0, 0] / (conf_matrix_model2[0, 0] + conf_matrix_model2[0, 1])
        f1_model2 = 2 * (precision_model2 * sensitivity_model2) / (precision_model2 + sensitivity_model2)

        fpr_model1, tpr_model1 = roc_curve((true_labels == '>50K').astype(int), model1_prob)
        fpr_model2, tpr_model2 = roc_curve((true_labels == '>50K').astype(int), model2_prob)

        roc_auc_model1 = auc(fpr_model1, tpr_model1)
        roc_auc_model2 = auc(fpr_model2, tpr_model2)

        global outputdata
        outputdata = f'Model 1:\nTrafność: {accuracy_model1:.5f}\nCzułość: {sensitivity_model1:.5f}\nSwoistość: {specificity_model1:.5f}\nPrecyzja: {precision_model1:.5f}\nWynik F1: {f1_model1:.5f}\nAUC: {roc_auc_model1:.2F}' \
                     f'\n\nModel 2:\nTrafność: {accuracy_model2:.5f}\nCzułość: {sensitivity_model2:.5f}\nSwoistość: {specificity_model2:.5f}\nPrecyzja: {precision_model2:.5f}\nWynik F1: {f1_model2:.5f}\nAUC: {roc_auc_model2:.2F}'\

        outputdata += '\n\nTrafność: Model 1 ' + compare(accuracy_model1, accuracy_model2) + ' Model 2\nCzułość: Model 1 ' + compare(sensitivity_model1, sensitivity_model2) + ' Model 2\nSwoistość: Model 1 ' + compare(specificity_model1, specificity_model2) + ' Model 2\nPrecyzja: Model 1 ' + compare(precision_model1, precision_model2) + ' Model 2\nWynik F1: Model 1 ' + compare(f1_model1, f1_model2) + ' Model 2\nAUC: Model 1 ' + compare(roc_auc_model1, roc_auc_model2) + ' Model 2'


        output_label.configure(text=output_label.cget("text") + outputdata)

        plt.plot(fpr_model1, tpr_model1, label='Model 1 (AUC = {:.2f})'.format(roc_auc_model1))
        plt.plot(fpr_model2, tpr_model2, label='Model 2 (AUC = {:.2f})'.format(roc_auc_model2))
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('Krzywa ROC')
        plt.legend()
        plt.show()
    else:
        true_values = data['rzeczywista'].astype(float)
        predictions_model1 = data['przewidywana1'].astype(float)
        predictions_model2 = data['przewidywana2'].astype(float)

        mae_model1, mape_model1, mse_model1, rmse_model1 = calculate_metrics(true_values, predictions_model1)
        mae_model2, mape_model2, mse_model2, rmse_model2 = calculate_metrics(true_values, predictions_model2)

        outputdata = f'\nModel 1:\nMAE: {mae_model1:.5f}\nMAPE: {mape_model1:.5f}\nMSE: {mse_model1:.5f}\nRMSE: {rmse_model1:.5f}' \
                     f'\n\nModel 2:\nMAE: {mae_model2:.5f}\nMAPE: {mape_model2:.5f}\nMSE: {mse_model2:.5f}\nRMSE: {rmse_model2:.5f}'

        outputdata += '\n\nMAE: Model 1 '  + compare(mae_model2, mae_model1) + ' Model 2 \nMAPE:  Model 1 '  + compare(mape_model2, mape_model1) + ' Model 2 \nMSE:  Model 1 '  + compare(mse_model2, mse_model1) + ' Model 2 \nRMSE:  Model 1 '  + compare(rmse_model2, rmse_model1) + ' Model 2 '

        output_label.configure(text=output_label.cget("text") + outputdata)

        errors_model1 = true_values - predictions_model1
        errors_model2 = true_values - predictions_model2

        plt.subplot(1, 2, 1)
        plt.hist(errors_model1, bins=20, density=True, alpha=0.6, color='b')
        plt.title('Histogram błędów - Model 1')
        plt.xlabel('Błąd')
        plt.ylabel('Częstość')

        plt.subplot(1, 2, 2)
        plt.hist(errors_model2, bins=20, density=True, alpha=0.6, color='r')
        plt.title('Histogram błędów - Model 2')
        plt.xlabel('Błąd')
        plt.ylabel('Częstość')

        xmin, xmax = min(errors_model1.min(), errors_model2.min()), max(errors_model1.max(), errors_model2.max())
        x = np.linspace(xmin, xmax, 100)
        plt.subplot(1, 2, 1)
        plt.plot(x, norm(x, np.mean(errors_model1), np.std(errors_model1)), 'k')
        plt.subplot(1, 2, 2)
        plt.plot(x, norm(x, np.mean(errors_model2), np.std(errors_model2)), 'k')

        plt.show()

root = tk.Tk()
root.title("Damian Kreński Klasyfikacja I Regresja")
plt.figure(figsize=(8, 6))

selected_option = tk.IntVar()
selected_option.set(1)
radio_button1 = tk.Radiobutton(root, text="Klasyfikacyjny", variable=selected_option, value=1)
radio_button1.pack(pady=5)
radio_button2 = tk.Radiobutton(root, text="Regresyjny", variable=selected_option, value=2)
radio_button2.pack(pady=5)
output_label = tk.Label(root, text="Wybranie pliku automatycznie uruchomi obliczenia. Wynik pojawi się poniżej tej linii.\nPodczas porówywania modeli znak > oznacza że model jest bardziej trafny/lepszy a nie zmienna jest większa. (analogiczne dla = i <)\n\n")
output_label.pack(pady=5)

button = tk.Button(root, text="Wybierz plik CSV", command=open_file_dialog)
button.pack(pady=20)

root.mainloop()