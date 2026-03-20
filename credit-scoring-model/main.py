from src.preprocess import load_and_preprocess
from src.train import train_model
from src.visualize import plot_target, plot_confusion_matrix

def main():
    path = "data/dataset.csv"

    print("📊 Loading Data...")
    data = load_and_preprocess(path)

    print("📈 Visualizing Data...")
    plot_target(data)

    print("🤖 Training Model...")
    X_test, y_test, y_pred = train_model(data)

    print("🎨 Showing Results...")
    plot_confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    main()