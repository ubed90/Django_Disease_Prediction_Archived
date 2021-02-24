# Symtoms Based Disease Prediction Using ML
A Webapp used for predicting Disease by taking input as symptoms which works using Machine Learning algorithms.

## INSTALLATION
<b>Make sure you have POSTGRES and PGADMIN4 installed on your computer</b><br>
<b>Go to shdps > disease_prediction > and change (name , user , passowrd) according to your own postgres server</b><br>
<b>Go to shdps > ModelAI > Right click on "Model" andc Copy it's path</b><br>
<b>After copying models path > Go to main_app > views.py > Line_16 and Paste the copied path replacing the filled one</b>

### THEN -->
> pip install virtualenv
> virtualenv env
> pip install django psycopg2 scikit-learn
> cd shdps/
> python manage.py makemigrations
> python manage.py migrate
> python manage.py runserver