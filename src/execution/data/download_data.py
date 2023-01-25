from datasets import load_dataset
import os


datasets = [
    'ddrg/named_math_formulas',
    'ddrg/math_formula_retrieval',
    'ddrg/math_text',
    'ddrg/math_formulas',
]


for ds_name in datasets:
    output = f"data/{ds_name}"
    if os.path.exists(output):
        print(f"Dataset {ds_name} already exists. Skipping...")
        continue

    print(f"Loading dataset {ds_name}...")
    ds = load_dataset(ds_name)
    os.makedirs(output, exist_ok=True)
    ds.save_to_disk(output)
    print(f"Dataset {ds_name} saved to {output}")