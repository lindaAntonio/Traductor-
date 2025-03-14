from datasets import Dataset

def load_opus_dataset(source_file, target_file, ids_file, max_samples=None):
    data = {"id": [], "source": [], "target": []}

    with open(source_file, encoding="utf-8") as src, open(target_file, encoding="utf-8") as tgt, open(ids_file, encoding="utf-8") as ids:
        for i, (id_line, src_line, tgt_line) in enumerate(zip(ids, src, tgt)):
            if max_samples and i >= max_samples:
                break
            data["id"].append(id_line.strip())
            data["source"].append(src_line.strip())
            data["target"].append(tgt_line.strip())
    
    return Dataset.from_dict(data)
