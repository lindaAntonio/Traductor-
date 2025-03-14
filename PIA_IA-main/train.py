import os
import torch
from transformers import (
    MBartForConditionalGeneration,
    MBart50Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from dataset import get_dataset
from utils import tokenize_function


def train_model(model_name="facebook/mbart-large-50-many-to-many-mmt", output_dir="./results/checkpoint-200"):

    # Verificar si existe un modelo entrenado
    if os.path.exists(output_dir):
        print("Modelo existente encontrado. Cargando para fine-tuning...")
        model = MBartForConditionalGeneration.from_pretrained(output_dir)
        tokenizer = MBart50Tokenizer.from_pretrained(output_dir)
    else:
        print("No se encontró un modelo entrenado. Iniciando desde el modelo preentrenado...")
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = MBart50Tokenizer.from_pretrained(model_name)

    # Obtener el dataset
    dataset = get_dataset()

    # Dividir el dataset en entrenamiento y validación (80% / 20%)
    dataset_split = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    # Tokenizar los datasets
    tokenized_train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    tokenized_eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )

    # Configurar el data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Configurar argumentos de entrenamiento
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,               # Directorio para guardar el modelo
        evaluation_strategy="epoch",        # Evaluar al final de cada época
        learning_rate=3e-5,                 # Tasa de aprendizaje
        per_device_train_batch_size=4,      # Tamaño del batch para entrenamiento
        per_device_eval_batch_size=4,       # Tamaño del batch para evaluación
        num_train_epochs=3,                 # Número de épocas
        gradient_accumulation_steps=2,      # Acumulación de gradientes
        weight_decay=0.01,                  # Decaimiento de peso
        save_total_limit=2,                 # Limitar checkpoints guardados
        save_strategy="epoch",              # Guardar al final de cada época
        predict_with_generate=True,         # Usar generación para evaluación
        fp16=torch.cuda.is_available(),     # Acelerar con FP16 si hay GPU
        dataloader_num_workers=4,           # Número de hilos para el dataloader
        logging_steps=100                   # Frecuencia de logs
    )

    # Crear entrenador
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,  # Cambiar `processing_class` por `tokenizer`
    )

    # Entrenar modelo
    print("Iniciando entrenamiento...")
    trainer.train()

    # Guardar modelo y tokenizador
    print(f"Guardando modelo entrenado en {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Entrenamiento finalizado.")
