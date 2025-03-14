# Traductor-
 Lo que se busca con este proyecto es que el usuario(s) pueda comunicarse de una manera
 entendible con las personas que no puedan hablar su mismo idioma, de tal forma que esto lo
 pueda ayudar a disfrutar de la experiencia de viajar.

 
 El proyecto en si es que a partir de una interfaz a la cual el usuario va tener acceso pueda
 ingresar una frase o palabra en el idioma que el necesita, y que al oprimir el botón de
 "traducir" pueda ver la traducción de la misma en el idioma que selecciono

 
  El proyecto consiste en desarrollar un sistema de traduccion automatica 
utilizando el modelo MBART, teniendo como finalidad desarrollar y entrenar un 
modelo de traduccion automatica, optimizando el desempeño y facilitando el 
uso mediante un interfaz de usuario simple.


 Objetivo principal: Entrenar y desplegar un modelo de traduccion automatica
 basado en el modelo MBART utilizando el dataset de OpenSubtitles.
 
 Objetivos secundarios:
 • Optimizar el tiempo de entrenamiento
 • Crear unainterfaz para facilitar la traduccion de textos

 
Codigo elaborado en python

 Frameworks principales:
 
 ● HuggngFace Transformers para la arquitectura MBART
 
 ● Datasets para la carga y manejo de datos
 ● Tkinter para interfaz grafica
 
 Recoleccion de datos se utilizaron fuentes de OpenSubtitles
 
 ● Usar eldataset de OpenSubtitles, ya dividido en archivos .en, .es, 
y .ids para las oraciones en inglés, español, y sus identificadores.

 ● Cargary tokenizar los datos


