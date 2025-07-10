class ReportGenerator:
    def __init__(self, class_names):
        self.class_names = class_names

    def generate_classification_report(self, y_true, y_pred):
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0)
        return report

    def plot_confusion_matrix(self, y_true, y_pred, normalize=True):
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

    def calculate_summary_metrics(self, y_true, y_pred):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        return metrics

    def print_summary(self, metrics):
        print("Evaluation Summary Metrics:")
        for key, value in metrics.items():
            print(f"  {key.capitalize()}: {value:.4f}")

