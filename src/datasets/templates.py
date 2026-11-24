# Adapted from:
#   https://github.com/openai/CLIP/blob/main/data/prompts.md (MIT)
#     - the per-dataset prompt sets
"""Zero-shot prompt templates, keyed by the dataset they were published for.

Each entry is a list of callables turning a class name into a prompt. The
wording is OpenAI's, published alongside CLIP; the grouping below simply
records which set goes with which dataset.
"""

PROMPTS = {
    "Cars": [
        lambda c: f"a photo of a {c}.",
        lambda c: f"a photo of the {c}.",
        lambda c: f"a photo of my {c}.",
        lambda c: f"i love my {c}!",
        lambda c: f"a photo of my dirty {c}.",
        lambda c: f"a photo of my clean {c}.",
        lambda c: f"a photo of my new {c}.",
        lambda c: f"a photo of my old {c}.",
    ],
    "CIFAR10": [
        lambda c: f"a photo of a {c}.",
        lambda c: f"a blurry photo of a {c}.",
        lambda c: f"a black and white photo of a {c}.",
        lambda c: f"a low contrast photo of a {c}.",
        lambda c: f"a high contrast photo of a {c}.",
        lambda c: f"a bad photo of a {c}.",
        lambda c: f"a good photo of a {c}.",
        lambda c: f"a photo of a small {c}.",
        lambda c: f"a photo of a big {c}.",
        lambda c: f"a photo of the {c}.",
        lambda c: f"a blurry photo of the {c}.",
        lambda c: f"a black and white photo of the {c}.",
        lambda c: f"a low contrast photo of the {c}.",
        lambda c: f"a high contrast photo of the {c}.",
        lambda c: f"a bad photo of the {c}.",
        lambda c: f"a good photo of the {c}.",
        lambda c: f"a photo of the small {c}.",
        lambda c: f"a photo of the big {c}.",
    ],
    "CIFAR100": [
        lambda c: f"a photo of a {c}.",
        lambda c: f"a blurry photo of a {c}.",
        lambda c: f"a black and white photo of a {c}.",
        lambda c: f"a low contrast photo of a {c}.",
        lambda c: f"a high contrast photo of a {c}.",
        lambda c: f"a bad photo of a {c}.",
        lambda c: f"a good photo of a {c}.",
        lambda c: f"a photo of a small {c}.",
        lambda c: f"a photo of a big {c}.",
        lambda c: f"a photo of the {c}.",
        lambda c: f"a blurry photo of the {c}.",
        lambda c: f"a black and white photo of the {c}.",
        lambda c: f"a low contrast photo of the {c}.",
        lambda c: f"a high contrast photo of the {c}.",
        lambda c: f"a bad photo of the {c}.",
        lambda c: f"a good photo of the {c}.",
        lambda c: f"a photo of the small {c}.",
        lambda c: f"a photo of the big {c}.",
    ],
    "DTD": [
        lambda c: f"a photo of a {c} texture.",
        lambda c: f"a photo of a {c} pattern.",
        lambda c: f"a photo of a {c} thing.",
        lambda c: f"a photo of a {c} object.",
        lambda c: f"a photo of the {c} texture.",
        lambda c: f"a photo of the {c} pattern.",
        lambda c: f"a photo of the {c} thing.",
        lambda c: f"a photo of the {c} object.",
    ],
    "EuroSAT": [
        lambda c: f"a centered satellite photo of {c}.",
        lambda c: f"a centered satellite photo of a {c}.",
        lambda c: f"a centered satellite photo of the {c}.",
    ],
    "Food101": [
        lambda c: f"a photo of {c}, a type of food.",
    ],
    "GTSRB": [
        lambda c: f'a zoomed in photo of a "{c}" traffic sign.',
        lambda c: f'a centered photo of a "{c}" traffic sign.',
        lambda c: f'a close up photo of a "{c}" traffic sign.',
    ],
    "MNIST": [
        lambda c: f'a photo of the number: "{c}".',
    ],
    "RESISC45": [
        lambda c: f"satellite imagery of {c}.",
        lambda c: f"aerial imagery of {c}.",
        lambda c: f"satellite photo of {c}.",
        lambda c: f"aerial photo of {c}.",
        lambda c: f"satellite view of {c}.",
        lambda c: f"aerial view of {c}.",
        lambda c: f"satellite imagery of a {c}.",
        lambda c: f"aerial imagery of a {c}.",
        lambda c: f"satellite photo of a {c}.",
        lambda c: f"aerial photo of a {c}.",
        lambda c: f"satellite view of a {c}.",
        lambda c: f"aerial view of a {c}.",
        lambda c: f"satellite imagery of the {c}.",
        lambda c: f"aerial imagery of the {c}.",
        lambda c: f"satellite photo of the {c}.",
        lambda c: f"aerial photo of the {c}.",
        lambda c: f"satellite view of the {c}.",
        lambda c: f"aerial view of the {c}.",
    ],
    "SUN397": [
        lambda c: f"a photo of a {c}.",
        lambda c: f"a photo of the {c}.",
    ],
    "SVHN": [
        lambda c: f'a photo of the number: "{c}".',
    ],
    "Flowers102": [
        lambda c: f"a photo of a {c}, a type of flower.",
    ],
    "FER2013": [
        lambda c: f"a photo of a {c} looking face.",
        lambda c: f"a photo of a face showing the emotion: {c}.",
        lambda c: f"a photo of a face looking {c}.",
        lambda c: f"a face that looks {c}.",
        lambda c: f"they look {c}.",
        lambda c: f"look at how {c} they are.",
    ],
    "PCAM": [
        lambda c: f"this is a photo of {c}",
    ],
    "OxfordIIITPet": [
        lambda c: f"a photo of a {c}, a type of pet.",
    ],
    "RenderedSST2": [
        lambda c: f"a {c} review of a movie.",
    ],
    "EMNIST": [
        lambda c: f'a photo of the number: "{c}".',
    ],
    "FashionMNIST": [
        lambda c: f"a photo of a {c}.",
        lambda c: f"a photo of the {c}.",
    ],
    "KMNIST": [
        lambda c: f"a photo of the character {c}.",
    ],
}

PROMPTS["STL10"] = PROMPTS["CIFAR10"]


def get_templates(dataset_name):
    """Prompt templates for ``dataset_name``.

    Validation variants reuse the templates of the dataset they are drawn from.
    """
    key = dataset_name[: -len("Val")] if dataset_name.endswith("Val") else dataset_name
    if key not in PROMPTS:
        raise ValueError(f"No prompt templates registered for dataset {dataset_name!r}.")
    return PROMPTS[key]
