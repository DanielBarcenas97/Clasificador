from flask import Flask,render_template,request,redirect,url_for,send_from_directory
import os

from clasificador import clasificadorReview
from clasificador import preprocesamiento_reviews

UPLOAD_FOLDER=os.path.abspath("./uploads")
#Aqui se van a guardar nuestros archivos 
app=Flask(__name__)
app.config["UPLOAD_FOLDER"]=UPLOAD_FOLDER
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/procesar', methods=['POST'])
def procesar():

    #Texto del input
    textingres = request.form.get("descripcion")
    #Metodo seleccionado
    boton = request.form["metodo"]

    


    if textingres=="":
        output = request.files["adjunto"]
        if output.filename == "":
            return render_template("resultado.html", resultado="Archivo no seleccionado")
        filename=output.filename
        output.save(os.path.join(app.config["UPLOAD_FOLDER"],filename))
        print("...............")
        direccion= os.path.join(app.config["UPLOAD_FOLDER"],filename)

        nombrePelicula = filename.replace(".txt","")
        nombrePelicula = nombrePelicula.replace("1.", "")
        nombrePelicula = nombrePelicula.replace("_", " ")
        archivo = open(direccion,'r')
        print("Entrenando modelo")
        review = archivo.read()
        archivo.close()
        txt = [review]
        data = preprocesamiento_reviews(txt)
        resul = data[0]

        res,exactitud = clasificadorReview(resul,boton)
        print(res[0])
        if res[0] == 1:
            clasificacion = "Buena Pelicula"
        else:
            clasificacion = "Mala Pelicula"

        return render_template("resultado.html", review=resul, metodo = "Metodo: "+ boton, classi = nombrePelicula + " es una "+ clasificacion, exactitud = "exactitud: " + exactitud)
    else:
        res,exactitud = clasificadorReview(textingres,boton)
        print(res[0])
        if res[0] == 1:
            clasificacion = "Buena Pelicula"
        else:
            clasificacion = "Mala Pelicula"
        
        return render_template("resultado.html", review=textingres, metodo = "Metodo: "+boton, classi = clasificacion, exactitud = "exactitud: " + exactitud)

@app.route('/cool_form', methods=['GET', 'POST'])
def cool_form():
    if request.method == 'POST':
        # do stuff when the form is submitted

        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('index'))

    # show the form, it wasn't submitted
    return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)