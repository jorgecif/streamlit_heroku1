# Aplicación para análisis exploratorio de datos y deploy en heroku

1.	Crear ventana nueva (vscode)
2.	Crear archivo app.py
3.	Crear archivo requirements.txt
a.	Ejecutar: pipreqs
4.	Crear archivo Procfile
a.	web: sh setup.sh && streamlit run app.py
5.	Crear archivo setup.sh
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
6.	Crear repositorio git local
Ejecutar: git init

7.	Login Heroku
a.	Heroku login
8.	Crear app Heroku
a.	Heroku créate
9.	Deploy
git add .
git commit -m "some message"
git push heroku master

10.	Check
heroku ps:scale web=1
11.	Open
Heroku open
