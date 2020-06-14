import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

'''
Proyecto Final Procesamiento inteligente de Textos
Integrantes:
Barcenas Martinez Edgar Daniel
Martinez Troncoso Julio Cesar
Silva Sandoval Cecilia
'''

'''
Se obtuvieron datos de 
https://ai.stanford.edu/~amaas/data/sentiment/

El dataset tiene 50,000 críticas de películas de estas la mitad son para 
entrenamiento el la otra para prueba. Cada parte tiene 12,500 criticas positivas
y 12,500 criticas negativas.

'''


'''
Metodo de limpieza de las review
Quitamos signos de puntuacion.
Etiquetas HTML y conversion del texto a minuscula
'''
def preprocesamiento_reviews(reviews):
    remplazar_sin_espacio = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
    remplazar_con_espacio = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    sin_espacio = ""
    espacio = " "
    reviews = [remplazar_sin_espacio.sub(sin_espacio, line.lower()) for line in reviews]
    reviews = [remplazar_con_espacio.sub(espacio, line) for line in reviews]
    return reviews

'''
Eliminamos palabras funcionales 
para mejorar el rendimiento de un modelo
'''
def eliminar_palabras_funcionales(corpus):
    palabras_funcionales = ['in', 'of', 'at', 'a', 'the']#stopwords.words('english')
    sin_palabras_funcionales = []
    for review in corpus:
        sin_palabras_funcionales.append(' '.join([palabra for palabra in review.split()
            if palabra not in palabras_funcionales]))
    return sin_palabras_funcionales


'''
Stemming
Normalización
'''
def get_stemmed_text(corpus):
    stemmer = PorterStemmer()
    return [' '.join([stemmer.stem(palabra) for palabra in review.split()]) for review in corpus]

'''
Lemmatization
Transformar la palabra en su raíz verdadera.
'''
def get_lemmatized_text(corpus):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(palabra) for palabra in review.split()]) for review in corpus]

def preprocesamiento_review(review):
    review = eliminar_palabras_funcionales(review)
    review = get_stemmed_text(review)
    review = get_lemmatized_text(review)
    return review

'''Lectura de datos'''
def leer_review_entrenamiento():
    reviews_train = []
    for line in open('movie_data/full_train.txt', 'r'):
        reviews_train.append(line.strip())
    return reviews_train

def leer_review_prueba():
    reviews_test = []
    for line in open('movie_data/full_test.txt', 'r'):
        reviews_test.append(line.strip())
    return reviews_test

def exactitud(X_train, X_val, y_train, y_val):
    for c in [0.001, 0.005, 0.01, 0.05, 0.1]:
        svm = LinearSVC(C=c)
        svm.fit(X_train, y_train)
        print ("Exactitud for C=%s: %s" 
           % (c, accuracy_score(y_val, svm.predict(X_val))))
    
def exactitudFinal(final,target,X,X_test):
    final.fit(X, target)
    print ("Final Exactitud: %s" 
       % accuracy_score(target, final.predict(X_test)))
    return accuracy_score(target, final.predict(X_test))

def prediccion(final,reviews_new,ngram_vectorizer):
    reviews_new_counts = ngram_vectorizer.transform(reviews_new)
    resultado = final.predict(reviews_new_counts)
    return resultado

def svmModificado(reviews_train_clean,reviews_test_clean):
    target = [1 if i < 12500 else 0 for i in range(25000)]
    stop_words = ['in', 'of', 'at', 'a', 'the']
    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words)
    ngram_vectorizer.fit(reviews_train_clean)
    X = ngram_vectorizer.transform(reviews_train_clean)
    X_test = ngram_vectorizer.transform(reviews_test_clean)
    X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)
    c=0.01
    final = LinearSVC(C=c)
    exactitud = exactitudFinal(final,target,X,X_test)
    return final,ngram_vectorizer,exactitud

'''
Se incluyen pares de palabras para tener mejor precisión
Este modelo probabilístico permite hacer una predicción estadística del próximo elemento.
'''
def modeloNgrams(reviews_train_clean,reviews_test_clean):
    target = [1 if i < 12500 else 0 for i in range(25000)]
    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
    ngram_vectorizer.fit(reviews_train_clean)
    X = ngram_vectorizer.transform(reviews_train_clean)
    X_test = ngram_vectorizer.transform(reviews_test_clean)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, target, train_size = 0.75)
   
    final_ngram = LogisticRegression(C=0.5)
    final_ngram.fit(X, target)
    exactitud = exactitudFinal(final_ngram,target,X,X_test)
    return final_ngram,ngram_vectorizer,exactitud
'''
Se utiliza la tecnica de recuento de palabras
para verificar si una palabra aparece mas de una vez y 
ayudar a determinar si esta es positiva o negativa
'''
def word_Counts(reviews_train_clean,reviews_test_clean):
    target = [1 if i < 12500 else 0 for i in range(25000)]
    wc_vectorizer = CountVectorizer(binary=False)
    wc_vectorizer.fit(reviews_train_clean)
    X = wc_vectorizer.transform(reviews_train_clean)
    X_test = wc_vectorizer.transform(reviews_test_clean)
    X_train, X_val, y_train, y_val = train_test_split(
        X, target, train_size = 0.75,)
    
    final_wc = LogisticRegression(C=0.05)
    final_wc.fit(X, target)
    exactitud = exactitudFinal(final_wc,target,X,X_test)
    return final_wc,wc_vectorizer,exactitud

'''
Término frecuencia de documento inversa de frecuencia:
Este representa la cantidad de veces que aparece una palabra especifica en la review.
'''
def TFIDF(reviews_train_clean,reviews_test_clean):
    target = [1 if i < 12500 else 0 for i in range(25000)]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(reviews_train_clean)
    X = tfidf_vectorizer.transform(reviews_train_clean)
    X_test = tfidf_vectorizer.transform(reviews_test_clean)
    X_train, X_val, y_train, y_val = train_test_split(
        X, target, train_size = 0.75)
    
    final_tfidf = LogisticRegression(C=1)
    final_tfidf.fit(X, target)
    exactitud = exactitudFinal(final_tfidf,target,X,X_test)
    # Final Accuracy: 0.882
    return final_tfidf,tfidf_vectorizer,exactitud

''' 
SVM + Ngrams nos da la mejor precisió del 90%
SVM clasificadol lineal con ngram_range=(1, 3)
'''
def svmModificado(reviews_train_clean,reviews_test_clean):
    target = [1 if i < 12500 else 0 for i in range(25000)]
    stop_words = ['in', 'of', 'at', 'a', 'the']
    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words)
    ngram_vectorizer.fit(reviews_train_clean)
    X = ngram_vectorizer.transform(reviews_train_clean)
    X_test = ngram_vectorizer.transform(reviews_test_clean)
    X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)
    c=0.01
    final = LinearSVC(C=c)
    exactitud =  exactitudFinal(final,target,X,X_test)
    return final,ngram_vectorizer,exactitud

'''SVM clasificadol lineal con ngram_range=(1, 2)'''
def svm(reviews_train_clean,reviews_test_clean):
    target = [1 if i < 12500 else 0 for i in range(25000)]
    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
    ngram_vectorizer.fit(reviews_train_clean)
    X = ngram_vectorizer.transform(reviews_train_clean)
    X_test = ngram_vectorizer.transform(reviews_test_clean)
    X_train, X_val, y_train, y_val = train_test_split(
        X, target, train_size = 0.75)
    final_svm_ngram = LinearSVC(C=0.01)
    final_svm_ngram.fit(X, target)
    exactitud =  exactitudFinal(final_svm_ngram,target,X,X_test)
    return final_svm_ngram, ngram_vectorizer,exactitud

'''
Naive_Bayes
Clasificamos en función de las probabilidades de las palabras.
'''
def Naive_Bayes(reviews_train_clean,reviews_test_clean):
    target = [1 if i < 12500 else 0 for i in range(25000)]
    stop_words = ['in', 'of', 'at', 'a', 'the']
    movie_vec = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words)
    movie_vec.fit(reviews_train_clean)
    X = movie_vec.transform(reviews_train_clean)
    X_test = movie_vec.transform(reviews_test_clean)
    X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75,)
    final_movie_vec = MultinomialNB()
    final_movie_vec.fit(X, target)
    exactitud = exactitudFinal(final_movie_vec,target,X,X_test)
    return final_movie_vec,movie_vec,exactitud

def cargarLimpiarReviews():
    reviews_train = leer_review_entrenamiento()
    reviews_test = leer_review_prueba()
    reviews_train_clean = preprocesamiento_reviews(reviews_train)
    reviews_test_clean = preprocesamiento_reviews(reviews_test)
    return reviews_train_clean,reviews_test_clean

def clasificadorReview(review,metodo):
    #cargar Datos
    reviews_train_clean,reviews_test_clean = cargarLimpiarReviews()

    if metodo == "SVM":
        modelo,vector,exactitud = svm(reviews_train_clean,reviews_test_clean)
        reviews_new = [review]
        resultado = prediccion(modelo,reviews_new,vector)
        
    elif metodo == "Naive_Bayes":
        modelo,vector,exactitud = Naive_Bayes(reviews_train_clean,reviews_test_clean)
        reviews_new = [review]
        resultado = prediccion(modelo,reviews_new,vector)

    elif metodo == "WordCounts":
        modelo,vector,exactitud = word_Counts(reviews_train_clean,reviews_test_clean)
        reviews_new = [review]
        resultado = prediccion(modelo,reviews_new,vector)

    elif metodo == "Ngrams":
        modelo,vector,exactitud = modeloNgrams(reviews_train_clean,reviews_test_clean)
        reviews_new = [review]
        resultado = prediccion(modelo,reviews_new,vector)
        
    elif metodo == "SVM+Ngrams":
        modelo,vector,exactitud = svmModificado(reviews_train_clean,reviews_test_clean)
        reviews_new = [review]
        resultado = prediccion(modelo,reviews_new,vector)
        
    elif metodo == "TFIDF":
        modelo,vector,exactitud = TFIDF(reviews_train_clean,reviews_test_clean)
        reviews_new = [review]
        resultado = prediccion(modelo,reviews_new,vector)
        
    return resultado,str(exactitud)


'''
    #Entreamos el modelo
    modelo,vector = Naive_Bayes(reviews_train_clean,reviews_test_clean)
    review = "Bad bad Bad Spider-Man: Far from Home stylishly sets the stage for the next era of the MCU."
    reviews_new = [review]
    resultado = prediccion(modelo,reviews_new,vector)
    print(resultado)

    #Entreamos el modelo
    modelo,vector = svm(reviews_train_clean,reviews_test_clean)
    review = "A breezily unpredictable blend of teen romance and superhero action, Spider-Man: Far from Home stylishly sets the stage for the next era of the MCU."
    reviews_new = [review]
    resultado = prediccion(modelo,reviews_new,vector)
    print(resultado)

    #Entreamos el modelo
    modelo,vector = svmModificado(reviews_train_clean,reviews_test_clean)
    reviews_new = [review]
    resultado = prediccion(modelo,reviews_new,vector)
    print(resultado)
    '''



'''
feature_to_coef = {
    word: coef for word, coef in zip(
        vector.get_feature_names(), modelo.coef_[0]
    )
}

for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:30]:
    print (best_positive)
    
print("")
for best_negative in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1])[:30]:
    print (best_negative)

'''
