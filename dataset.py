from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
import albumentations
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch


def build_dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """
    Build the dataset for object detection.

    Returns:
        The dataset.

    Below is an example of how to load an object detection dataset.

    ```python
    from datasets import load_dataset

    raw_datasets = load_dataset("cppe-5")
    if "validation" not in dataset_base:
        split = dataset_base["train"].train_test_split(0.15, seed=1337)
        dataset_base["train"] = split["train"]
        dataset_base["validation"] = split["test"]
    ```

    Ref: https://huggingface.co/docs/datasets/v3.2.0/package_reference/main_classes.html#datasets.DatasetDict

    You can replace this with your own dataset. Make sure to include
    the `test` split and ensure that it is consistent with the dataset format expected for object detection.
    For example:
        raw_datasets["test"] = load_dataset("cppe-5", split="test")
    """
    # Write your code here.
    from datasets import load_dataset
    all_datasets = load_dataset("cppe-5")  # Load the dataset
    # Divide the training set into an 80% training set and a 20% validation set.
    train_dataset = all_datasets["train"]
    train_val_split = train_dataset.train_test_split(test_size=0.2, seed=42)
    all_datasets["train"] = train_val_split["train"]
    all_datasets["validation"] = train_val_split["test"]
    return all_datasets


# Data enhancement configuration
train_augmentation = albumentations.Compose([
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.Rotate(limit=10, p=0.5),
    albumentations.RandomBrightnessContrast(p=0.5),
    albumentations.Resize(height=640, width=640),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

validation_test_augmentation = albumentations.Compose([
    albumentations.Resize(height=640, width=640),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])  # For validation sets and test sets


# Define preprocessor functions
def preprocess_train(example, processor):
    if not example:
        return None
    image = example["image"]
    bboxes = example.get("bbox", [])
    category_labels = example.get("category_labels", [])

    # Check if image is a batch of images
    if len(image.shape) == 4:
        image = image[0]  # Take the first image in the batch

    # Convert the image to BGR format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Apply data enhancement
    transformed = train_augmentation(image=image, bboxes=bboxes, category_ids=category_labels)
    image = transformed["image"]
    bboxes = transformed["bboxes"]
    category_labels = transformed["category_ids"]

    # Converts category labels to tensors
    labels = [{"labels": torch.tensor(category_labels), "boxes": torch.tensor(bboxes)}]

    # Use a processor for preprocessing
    inputs = processor(images=image, return_tensors="pt")
    return {**inputs, "labels": labels}


def preprocess_val_test(example, processor):
    if not example:
        return None
    image = example["image"]
    bboxes = example.get("bbox", [])
    category_labels = example.get("category_labels", [])

    # Check if image is a batch of images
    if len(image.shape) == 4:
        image = image[0]  # Take the first image in the batch

    # Convert the image to BGR format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # # Apply validation set enhancement
    transformed = validation_test_augmentation(image=image, bboxes=bboxes, category_ids=category_labels)
    image = transformed["image"]
    bboxes = transformed["bboxes"]
    category_labels = transformed["category_ids"]

    # Converts category labels to tensors
    labels = [{"labels": torch.tensor(category_labels), "boxes": torch.tensor(bboxes)}]

    # Use a processor for preprocessing
    inputs = processor(images=image, return_tensors="pt")
    return {**inputs, "labels": labels}


def add_preprocessing(dataset, processor) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """
    Add preprocessing to the dataset.

    Args:
        dataset: The dataset to preprocess.
        processor: The image processor to use for preprocessing.

    Returns:
        The preprocessed dataset.

    In this function, you can add any preprocessing steps to the dataset.
    For example, you can add data augmentation, normalization or formatting to meet the model input, etc.

    Hint:
    # You can use the `with_transform` method of the dataset to apply transformations.
    # Ref: https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.Dataset.with_transform

    # You can also use the `map` method of the dataset to apply transformations.
    # Ref: https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/main_classes#datasets.Dataset.map

    # For Augmentation, you can use the `albumentations` library.
    # Ref: https://albumentations.ai/docs/

    from functools import partial

    # Create the batch transform functions for training and validation sets
    train_transform_batch = # Callable for train set transforming with batched samples passed
    validation_transform_batch = # Callable for val/test set transforming with batched samples passed

    # Apply transformations to dataset splits
    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(validation_transform_batch)
    dataset["test"] = dataset["test"].with_transform(validation_transform_batch)
    """
    # Write your code here.
    from functools import partial
    # Create batch transform function
    train_transform_batch = partial(preprocess_train, processor=processor)
    validation_test_transform_batch = partial(preprocess_val_test, processor=processor)

    # Application transformation
    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(validation_test_transform_batch)
    dataset["test"] = dataset["test"].with_transform(validation_test_transform_batch)

    return dataset
