import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
import time

def load_and_preprocess_dataset(
    test_dir: str,
    image_size: tuple,
    batch_size: int,
    num_classes: int
) -> tuple:
    """Load and preprocess the test dataset."""
    ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )
    
    class_names = ds.class_names
    
    ds = ds.map(lambda x, y: (
        tf.cast(x, tf.float32) / 127.5 - 1,
        tf.one_hot(y, num_classes)
    ))
    
    return ds, class_names

def evaluate_model(model_path: str, test_dir: str):
    """Evaluate model and print metrics."""
    # Configuration
    HEIGHT = 48
    WIDTH = int(HEIGHT * (320/240))
    IMAGE_SIZE = (HEIGHT, WIDTH)
    NUM_CLASSES = 12
    BATCH_SIZE = 32
    
    print(f"\nLoading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print("Loading and preprocessing test dataset...")
    test_ds, class_names = load_and_preprocess_dataset(
        test_dir,
        IMAGE_SIZE,
        BATCH_SIZE,
        NUM_CLASSES
    )
    
    print("\nEvaluating model...")
    start_time = time.time()
    
    # Get model predictions
    predictions = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    
    # Get true labels
    y_true = []
    for _, labels in test_ds:
        y_true.extend(np.argmax(labels, axis=1))
    
    # Calculate metrics
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    
    # Generate classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=3
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nDetailed Classification Report:")
    print(report)
    
    evaluation_time = time.time() - start_time
    print(f"\nEvaluation completed in {evaluation_time:.2f} seconds")
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'predictions': predictions,
        'true_labels': y_true,
        'class_names': class_names
    }

if __name__ == "__main__":
    MODEL_PATH = "Efficient_CNN_Model_final.keras"
    TEST_DIR = "thermal/Dataset-20241124T163610Z-001/Dataset/test"
    
    results = evaluate_model(MODEL_PATH, TEST_DIR)
