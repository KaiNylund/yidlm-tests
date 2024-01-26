from datasets import load_dataset, Dataset, DatasetDict
from functools import partial
from tqdm import tqdm

SEED = 42
LANG_SPLIT_SIZE = 10000

def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds

langs_with_10k_docs = [
    "af", "sq", "am", "ar", "hy", "as", "az", "bn", "ba", "eu", "be", "br", "bg",
    "my", "ca", "ceb", "ckb", "ce", "zh", "cv", "hr", "cs", "da", "dv", "nl", "en",
    "eo", "et", "tl", "fi", "fr", "gl", "ka", "de", "el", "gu", "he", "hi", "hu",
    "is", "id", "ga", "it", "ja", "kn", "kk", "km", "ko", "ku", "ky", "lo", "la",
    "lv", "lt", "lb", "mk", "mg", "ms", "ml", "mr", "mn", "ne", "no", "nn", "or",
    "ps", "fa", "pl", "pt", "pa", "ro", "ru", "sah", "sa", "sr", "sd", "si", "sk",
    "sl", "azb", "es", "sv", "tg", "ta", "tt", "te", "th", "bo", "tr", "uk", "ur",
    "ug", "uz", "vi", "cy", "fy", "pnb", "yi"
]

for lang_id in tqdm(langs_with_10k_docs):
    i_dataset = load_dataset('oscar-corpus/OSCAR-2301', lang_id, split='train', streaming=True)
    i_dataset_split = i_dataset.take(LANG_SPLIT_SIZE)

    ds = Dataset.from_generator(partial(gen_from_iterable_dataset, i_dataset_split),
                                features=i_dataset_split.features)
    # Double check there are enough docs
    if ds.num_rows == LANG_SPLIT_SIZE:
        ds = ds.shuffle(seed=SEED)
        ds_traintest = ds.train_test_split(test_size=0.2)
        ds_testvalid = ds_traintest["test"].train_test_split(test_size=0.5)
        ds = DatasetDict({
            "train": ds_traintest["train"],
            "test": ds_testvalid["test"],
            "dev": ds_testvalid["train"]
        })
        ds.save_to_disk(f"./lang_splits/{lang_id}/")