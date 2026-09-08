import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from classifier import classify_waste


DATASET_DIR = "dataset"


def evaluate_model():

    true_labels = []
    predicted_labels = []

    print("\n♻️ Smart Waste AI Evaluation")
    print("=" * 50)

    if not os.path.exists(DATASET_DIR):
        print("❌ Dataset folder not found.")
        return

    for category in sorted(os.listdir(DATASET_DIR)):

        category_path = os.path.join(
            DATASET_DIR,
            category
        )

        if not os.path.isdir(category_path):
            continue

        print(f"\nTesting category: {category}")

        for filename in sorted(os.listdir(category_path)):

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            image_path = os.path.join(
                category_path,
                filename
            )

            try:

                result = classify_waste(image_path)

                predicted = result["waste_type"].replace(
                    " ",
                    "_"
                )

                true_labels.append(category)
                predicted_labels.append(predicted)

                print(
                    f"{filename}: "
                    f"{predicted} "
                    f"({result['confidence'] * 100:.2f}%)"
                )

            except Exception as error:

                print(
                    f"❌ Error processing {filename}: "
                    f"{error}"
                )

    if not true_labels:

        print("\n❌ No test images found.")
        print("Add images to the dataset folders first.")
        return

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0
    )

    print("\n")
    print("=" * 50)
    print("📊 MODEL EVALUATION RESULTS")
    print("=" * 50)

    print(f"Accuracy :  {accuracy * 100:.2f}%")
    print(f"Precision:  {precision * 100:.2f}%")
    print(f"Recall   :  {recall * 100:.2f}%")
    print(f"F1 Score :  {f1 * 100:.2f}%")

    print("\n📋 Classification Report")
    print("=" * 50)

    print(
        classification_report(
            true_labels,
            predicted_labels,
            zero_division=0
        )
    )


if __name__ == "__main__":
    evaluate_model()
