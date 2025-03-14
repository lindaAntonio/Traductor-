from train import train_model
from translate import translate_text
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def main():
    try:
       
        entrenar = input("¿Deseas entrenar el modelo? (s/n): ").lower()

        if entrenar == "s":
            print("Entrenando el modelo...")
            train_model()

        source_text = "You only fail when you stop trying"
        traduccion_manual= "Solo fallas cuando dejas de intentarlo"
        translated_text = translate_text(source_text, source_lang="en_XX", target_lang="es_XX")
        
        print("Texto original:", source_text)
        print("Texto traducido:", translated_text[0])

        

        bleu = calcular_bleu(traduccion_manual, translated_text[0])

        print(f"Puntaje BLEU: {bleu:.2f}")

    except Exception as e:
        print(f"Ocurrió un error: {e}")


def calcular_bleu(referencia, prediccion):

    referencia_tokens = [referencia.split()]
    prediccion_tokens = prediccion.split()
    
    smoothing = SmoothingFunction().method4
    
    # Calcular BLEU
    bleu_score = sentence_bleu(referencia_tokens, prediccion_tokens, smoothing_function=smoothing)
    return bleu_score


if __name__ == "__main__":
    main()
