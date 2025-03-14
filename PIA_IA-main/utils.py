def tokenize_function(examples, tokenizer, max_length=512):
    """
    Tokeniza los datos de entrada y salida.
    """
    inputs = examples["source"]  # Cambiar 'source' por el nombre correcto en tu dataset.
    targets = examples["target"]
    model_inputs = tokenizer(inputs, padding=True, truncation=True, max_length=max_length)
    labels = tokenizer(targets, padding=True, truncation=True, max_length=max_length).input_ids
    model_inputs["labels"] = labels
    return model_inputs
