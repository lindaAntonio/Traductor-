from preprocess import load_opus_dataset
#Menor samples para agilizar training , originalmente eran 60 000
def get_dataset(max_samples=1000):
    source_file = "data/OpenSubtitles.en-es.en"
    target_file = "data/OpenSubtitles.en-es.es"
    ids_file = "data/OpenSubtitles.en-es.ids"
    return load_opus_dataset(source_file, target_file, ids_file, max_samples=max_samples)
