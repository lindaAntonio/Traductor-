import os
import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer

def translate_text(text, source_lang, target_lang, model_dir="./results/checkpoint-200"):
    """
    Traduce texto usando un modelo MBART entrenado o preentrenado.
    """
    # Verificar si existe un modelo entrenado
    if os.path.exists(model_dir):
        print(f"Cargando modelo entrenado desde {model_dir}...")
        model = MBartForConditionalGeneration.from_pretrained(model_dir)
        tokenizer = MBart50Tokenizer.from_pretrained(model_dir)

    else:
        raise FileNotFoundError(f"No se encontró un modelo entrenado en {model_dir}. Por favor, entrena el modelo primero.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Ajuste de gpu o cpu
    model.to(device)
    model.config.src_lang = source_lang
    model.config.tgt_lang = target_lang

    tokenizer.src_lang = source_lang
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    # Generar traducción
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_length=200,
            num_beams=5,  # Usa un valor menor para depurar
            early_stopping=True,
            forced_bos_token_id=tokenizer.lang_code_to_id[source_lang]   #Cambiar a tar lng
        )

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
