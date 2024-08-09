from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def plot_ROC(y_predicted, y_observed, title='Receiver operating characteristic'):
    '''
        This function is an adaptation from `Hands-On Machine Learning
        with Scikit-Learn & TensorFlow` Aurélien Géron. (page 93).
    '''
    positive_class_scores_test = y_predicted
    is_positive_class_test = y_observed == 1

    fpr, tpr, thresholds = roc_curve(is_positive_class_test,
                                     positive_class_scores_test)

    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange',
            lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    
    return fig