import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_target(data):
    # ✅ Create folder automatically
    os.makedirs("outputs/plots", exist_ok=True)

    plt.figure()
    sns.countplot(x=data["Risk"])
    plt.title("Credit Risk Distribution")

    plt.savefig("outputs/plots/target_distribution.png")
    plt.show()


def plot_confusion_matrix(y_test, y_pred):
    from sklearn.metrics import confusion_matrix

    # ✅ Create folder automatically
    os.makedirs("outputs/plots", exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("outputs/plots/confusion_matrix.png")
    plt.show()