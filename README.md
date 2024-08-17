# ISY503_Assesment_3
NLP Project to provide the sentiment analysis of inputted text. 

Project autors:
Aleksandr Andrianov A00141332

Datasets were taken from:
https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html

To run the app on local pc:

1. python3 install django
2. python3 manage.py runserver
3. Open browser on http://127.0.0.1:8000/

To make changes in model.py file:

1. Modify model files
2. Delete all files in migrations, delete database, delete python cache
3. python manage.py makemigrations nlp_web_app
4. python3 manage.py migrate
