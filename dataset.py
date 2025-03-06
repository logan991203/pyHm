from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
import albumentations
import numpy as np


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

    # Delete the data with the boundary box problem
    remove_idx = [590, 821, 822, 875, 876, 878, 879]
    keep = [i for i in range(len(all_datasets["train"])) if i not in remove_idx]
    all_datasets["train"] = all_datasets["train"].select(keep)

    # Divide the training set into an 80% training set and a 20% validation set.
    train_dataset = all_datasets["train"]
    train_val_split = train_dataset.train_test_split(test_size=0.2, seed=42)
    all_datasets["train"] = train_val_split["train"]
    all_datasets["validation"] = train_val_split["test"]
    return all_datasets


# Data enhancement configuration
train_augmentation = albumentations.Compose(
    [
        albumentations.HorizontalFlip(p=1.0),
        albumentations.VerticalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(p=1.0),
        albumentations.Resize(height=480, width=480),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]))  # For training set

validation_test_augmentation = albumentations.Compose(
    [
        albumentations.Resize(height=480, width=480),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=[
        "category"]))  # For training set)  # For validation sets and test sets


# # Formatting comments
def formatted_com(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
    annotations.append(new_ann)
    return annotations


# transforming a batch
def train_transform_batch(examples, processor):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = train_augmentation(image=image, bboxes=objects["bbox"], category=objects["category"])
        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])
    targets = [{"image_id": id_, "annotations": formatted_com(id_, cat_, ar_, box_)}
               for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)]
    return processor(images=images, annotations=targets, return_tensors="pt")


def val_test_transform_batch(examples, processor):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = validation_test_augmentation(image=image, bboxes=objects["bbox"], category=objects["category"])
        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])
    targets = [{"image_id": id_, "annotations": formatted_com(id_, cat_, ar_, box_)}
               for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)]
    return processor(images=images, annotations=targets, return_tensors="pt")


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
    train_set_transform_batch = partial(train_transform_batch, processor=processor)
    val_test_set_transform_batch = partial(val_test_transform_batch, processor=processor)
    # Application transformation
    dataset["train"] = dataset["train"].with_transform(train_set_transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(val_test_set_transform_batch)
    dataset["test"] = dataset["test"].with_transform(val_test_set_transform_batch)

    return dataset
