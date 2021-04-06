from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http import JsonResponse
from datetime import date
import os , re
import pandas as pd
import joblib as jb
from sklearn.preprocessing import LabelEncoder
from django.contrib import messages
from django.contrib.auth.models import User , auth
from .models import patient , doctor , diseaseinfo , consultation ,rating_review
from chats.models import Chat,Feedback

# Create your views here.


# Loading DataSet
encoder = LabelEncoder()
dirs = os.listdir('E:\PROGRAMMING\Projects\Django_Projects\Django_Disease_Prediction\SHDPS_MODEL_SELECTION\Datasets')
dirs = dirs[2]
num = int(re.findall('[0-9]+' , dirs)[0])
headers = [*pd.read_csv(f'E:\PROGRAMMING\Projects\Django_Projects\Django_Disease_Prediction\SHDPS_MODEL_SELECTION\Datasets\SHDPS_Training_{num}.csv', nrows=1)]
dataset = pd.read_csv(f'E:\PROGRAMMING\Projects\Django_Projects\Django_Disease_Prediction\SHDPS_MODEL_SELECTION\Datasets\SHDPS_Training_{num}.csv', usecols=[c for c in headers if c != 'Unnamed: 0'])

#loading model
# model = jb.load(r'E:\PROGRAMMING\Projects\Django_Disease_Prediction_System\shdps\ModelAI\model')

def get_disease(model , user_symptoms , predicted , Y_train , Y):
   for num , disease in zip(Y_train , Y):
            if int(num) == int(predicted[0]):  
               predicted = disease
               cs = model.predict_proba(user_symptoms)
               cs = cs.max()*100
               return predicted , cs
   else:
      return None , None


def get_disease_with_max_accuracy(user_symptoms , Y_train , Y):
   disease_and_Cs = []
   dt_model = jb.load('E:\PROGRAMMING\Projects\Django_Projects\Django_Disease_Prediction\SHDPS_MODEL_SELECTION\Decision_Tree\Decision_Tree_Model.sav')
   knn_model = jb.load('E:\PROGRAMMING\Projects\Django_Projects\Django_Disease_Prediction\SHDPS_MODEL_SELECTION\K_Nearest_Neighbors\KNN_Model.sav')
   svm_model = jb.load('E:\PROGRAMMING\Projects\Django_Projects\Django_Disease_Prediction\SHDPS_MODEL_SELECTION\Kernel_SVM\Kernel_SVM_Model.sav')
   lr_model = jb.load('E:\PROGRAMMING\Projects\Django_Projects\Django_Disease_Prediction\SHDPS_MODEL_SELECTION\Logistic_Regression\Logistic_Regression_Model.sav')
   gaussian_NB_model = jb.load(r'E:\PROGRAMMING\Projects\Django_Projects\Django_Disease_Prediction\SHDPS_MODEL_SELECTION\Naive_Bayes\Gaussian_Bayes_Model.sav')
   multinomial_NB_model = jb.load(r'E:\PROGRAMMING\Projects\Django_Projects\Django_Disease_Prediction\SHDPS_MODEL_SELECTION\Naive_Bayes\Multinomial_Bayes_Model.sav')
   rf_model = jb.load('E:\PROGRAMMING\Projects\Django_Projects\Django_Disease_Prediction\SHDPS_MODEL_SELECTION\Random_Forest\Random_Forest_Model.sav')

   dt_model_predicted = dt_model.predict(user_symptoms)
   dt_model_predicted_disease , dt_model_predicted_CS = get_disease(dt_model , user_symptoms , dt_model_predicted , Y_train , Y)
   disease_and_Cs.append((dt_model_predicted_disease , dt_model_predicted_CS))

   knn_model_predicted = knn_model.predict(user_symptoms)
   knn_model_predicted_disease , knn_model_predicted_CS = get_disease(knn_model , user_symptoms , knn_model_predicted , Y_train , Y)
   disease_and_Cs.append((knn_model_predicted_disease , knn_model_predicted_CS))

   svm_model_predicted = svm_model.predict(user_symptoms)
   svm_model_predicted_disease , svm_model_predicted_CS = get_disease(svm_model , user_symptoms , svm_model_predicted , Y_train , Y)
   disease_and_Cs.append((svm_model_predicted_disease , svm_model_predicted_CS))

   lr_model_predicted = lr_model.predict(user_symptoms)
   lr_model_predicted_disease , lr_model_predicted_CS = get_disease(lr_model , user_symptoms , lr_model_predicted , Y_train , Y)
   disease_and_Cs.append((lr_model_predicted_disease , lr_model_predicted_CS))

   gaussian_NB_model_predicted = gaussian_NB_model.predict(user_symptoms)
   gaussian_NB_model_predicted_disease , gaussian_NB_model_predicted_CS = get_disease(gaussian_NB_model , user_symptoms , gaussian_NB_model_predicted , Y_train , Y)
   disease_and_Cs.append((gaussian_NB_model_predicted_disease , gaussian_NB_model_predicted_CS))

   multinomial_NB_model_predicted = multinomial_NB_model.predict(user_symptoms)
   multinomial_NB_model_predicted_disease , multinomial_NB_model_predicted_CS = get_disease(multinomial_NB_model , user_symptoms , multinomial_NB_model_predicted , Y_train , Y)
   disease_and_Cs.append((multinomial_NB_model_predicted_disease , multinomial_NB_model_predicted_CS))

   rf_model_predicted = rf_model.predict(user_symptoms)
   rf_model_predicted_disease , rf_model_predicted_CS = get_disease(rf_model , user_symptoms , rf_model_predicted , Y_train , Y)
   disease_and_Cs.append((rf_model_predicted_disease , rf_model_predicted_CS))

   disease_and_Cs.sort(key = lambda x: x[1] , reverse=True)
   return disease_and_Cs[0][0] , disease_and_Cs[0][1]

   


def home(request):
   puser = None
   duser = None
   r = None
   try:
      patientusername = request.session['patientusername']
      puser = User.objects.get(username=patientusername)
      doctorusername = request.session['doctorusername']
      duser = User.objects.get(username=doctorusername)
      r = rating_review.objects.filter(doctor=duser.doctor)
   except:
      pass
   print(duser , puser)
   return render(request, 'index.html' , {'puser' : puser , 'duser' : duser , 'rate' : r})


def contact(request):
   puser = None
   duser = None
   r = None
   try:
      patientusername = request.session['patientusername']
      puser = User.objects.get(username=patientusername)
      doctorusername = request.session['doctorusername']
      duser = User.objects.get(username=doctorusername)
      r = rating_review.objects.filter(doctor=duser.doctor)
   except:
      pass
   print(duser , puser)
   return render(request, 'contact.html' , {'puser' : puser , 'duser' : duser , 'rate' : r})


def about(request):
   puser = None
   duser = None
   r = None
   try:
      patientusername = request.session['patientusername']
      puser = User.objects.get(username=patientusername)
      doctorusername = request.session['doctorusername']
      duser = User.objects.get(username=doctorusername)
      r = rating_review.objects.filter(doctor=duser.doctor)
   except:
      pass
   print(duser , puser)
   return render(request, 'about.html' , {'puser' : puser , 'duser' : duser , 'rate' : r})
   

       


def admin_ui(request):

    if request.method == 'GET':

      if request.user.is_authenticated:

        auser = request.user
        Feedbackobj = Feedback.objects.all()

        return render(request,'admin/admin_ui/admin_ui.html' , {"auser":auser,"Feedback":Feedbackobj})

      else :
        return redirect('home')



    if request.method == 'POST':

       return render(request,'patient/patient_ui/profile.html')





def patient_ui(request):

    if request.method == 'GET':

      if request.user.is_authenticated:

        patientusername = request.session['patientusername']
        puser = User.objects.get(username=patientusername)

        return render(request,'patient/patient_ui/profile.html' , {"puser":puser})

      else :
        return redirect('home')



    if request.method == 'POST':

       return render(request,'patient/patient_ui/profile.html')

       


def pviewprofile(request, patientusername):

    if request.method == 'GET':

          puser = User.objects.get(username=patientusername)

          return render(request,'patient/view_profile/view_profile.html', {"puser":puser})




def checkdisease(request):
  patientusername = request.session['patientusername']
  puser = User.objects.get(username=patientusername)
  Y = dataset.iloc[ : , 0].values
  Y_train = encoder.fit_transform(Y)

  symptomslist = list(dataset.columns[1 : ])

  alphabaticsymptomslist = sorted(symptomslist)

  


  if request.method == 'GET':
    
     return render(request,'patient/checkdisease/checkdisease.html', {"list2":alphabaticsymptomslist , 'puser':puser})




  elif request.method == 'POST':
       
      ## access you data by playing around with the request.POST object
      
      inputno = int(request.POST["noofsym"])
      print(inputno)
      if (inputno == 0 ) :
          return JsonResponse({'predicteddisease': "none",'confidencescore': 0 })
  
      else :

        psymptoms = []
        psymptoms = request.POST.getlist("symptoms[]")
       
        print(psymptoms)

      
        """      #main code start from here...
        """
      

      
        testingsymptoms = []
        #append zero in all coloumn fields...
        for x in range(0, len(symptomslist)):
          testingsymptoms.append(0)


        #update 1 where symptoms gets matched...
        for k in range(0, len(symptomslist)):

          for z in psymptoms:
              if (z == symptomslist[k]):
                  testingsymptoms[k] = 1


        inputtest = [testingsymptoms]

        print(inputtest)
      
        predicted , confidencescore = get_disease_with_max_accuracy(inputtest , Y_train , Y)
        print("predicted disease is : ")
        print(predicted)

        confidencescore = format(confidencescore, '.0f')
        print(confidencescore)
        predicted_disease = predicted

        

        #consult_doctor codes----------

        #   doctor_specialization = ["Rheumatologist","Cardiologist","ENT specialist","Orthopedist","Neurologist",
        #                             "Allergist/Immunologist","Urologist","Dermatologist","Gastroenterologist"]
        

        Rheumatologist = [  'Osteoarthristis','Arthritis']
       
        Cardiologist = [ 'Heart attack','Bronchial Asthma','Hypertension ']
       
        ENT_specialist = ['(vertigo) Paroymsal  Positional Vertigo','Hypothyroidism' ]

        Orthopedist = []

        Neurologist = ['Varicose veins','Paralysis (brain hemorrhage)','Migraine','Cervical spondylosis']

        Allergist_Immunologist = ['Allergy','Pneumonia',
        'AIDS','Common Cold','Tuberculosis','Malaria','Dengue','Typhoid']

        Urologist = [ 'Urinary tract infection',
         'Dimorphic hemmorhoids(piles)']

        Dermatologist = [  'Acne','Chicken pox','Fungal infection','Psoriasis','Impetigo']

        Gastroenterologist = ['Peptic ulcer diseae', 'GERD','Chronic cholestasis','Drug Reaction','Gastroenteritis','Hepatitis E',
        'Alcoholic hepatitis','Jaundice','hepatitis A',
         'Hepatitis B', 'Hepatitis C', 'Hepatitis D','Diabetes ','Hypoglycemia']
         
        if predicted_disease in Rheumatologist :
           consultdoctor = "Rheumatologist"
           
        if predicted_disease in Cardiologist :
           consultdoctor = "Cardiologist"
           

        elif predicted_disease in ENT_specialist :
           consultdoctor = "ENT specialist"
     
        elif predicted_disease in Orthopedist :
           consultdoctor = "Orthopedist"
     
        elif predicted_disease in Neurologist :
           consultdoctor = "Neurologist"
     
        elif predicted_disease in Allergist_Immunologist :
           consultdoctor = "Allergist/Immunologist"
     
        elif predicted_disease in Urologist :
           consultdoctor = "Urologist"
     
        elif predicted_disease in Dermatologist :
           consultdoctor = "Dermatologist"
     
        elif predicted_disease in Gastroenterologist :
           consultdoctor = "Gastroenterologist"
     
        else :
           consultdoctor = "other"


        request.session['doctortype'] = consultdoctor 

     

        #saving to database.....................

        patient = puser.patient
        diseasename = predicted_disease
        no_of_symp = inputno
        symptomsname = psymptoms
        confidence = confidencescore

        diseaseinfo_new = diseaseinfo(patient=patient,diseasename=diseasename,no_of_symp=no_of_symp,symptomsname=symptomsname,confidence=confidence,consultdoctor=consultdoctor)
        diseaseinfo_new.save()
        

        request.session['diseaseinfo_id'] = diseaseinfo_new.id

        print("disease record saved sucessfully.............................")

        return JsonResponse({'predicteddisease': predicted_disease ,'confidencescore':confidencescore , "consultdoctor": consultdoctor})
   


   
    



   





def pconsultation_history(request):

    if request.method == 'GET':

      patientusername = request.session['patientusername']
      puser = User.objects.get(username=patientusername)
      patient_obj = puser.patient
        
      consultationnew = consultation.objects.filter(patient = patient_obj)
      
    
      return render(request,'patient/consultation_history/consultation_history.html',{"consultation":consultationnew , 'puser' : puser})


def dconsultation_history(request):

    if request.method == 'GET':

      doctorusername = request.session['doctorusername']
      duser = User.objects.get(username=doctorusername)
      doctor_obj = duser.doctor
        
      consultationnew = consultation.objects.filter(doctor = doctor_obj)
      
    
      return render(request,'doctor/consultation_history/consultation_history.html',{"consultation":consultationnew})



def doctor_ui(request):

    if request.method == 'GET':

      doctorid = request.session['doctorusername']
      duser = User.objects.get(username=doctorid)
      print(duser)
    
      return render(request,'doctor/doctor_ui/profile.html',{"duser":duser})



      


def dviewprofile(request, doctorusername):

    if request.method == 'GET':
         puser = None
         try:
            patientusername = request.session['patientusername']
            puser = User.objects.get(username=patientusername)
         except:
            pass
         duser = User.objects.get(username=doctorusername)
         print(duser , puser)
         r = rating_review.objects.filter(doctor=duser.doctor)
       
         return render(request,'doctor/view_profile/view_profile.html', {"duser":duser, "puser" : puser , "rate":r} )








       
def  consult_a_doctor(request):


    if request.method == 'GET':

        patientusername = request.session['patientusername']
        puser = User.objects.get(username=patientusername)
        doctortype = request.session['doctortype']
        print(doctortype)
        dobj = doctor.objects.all()
        #dobj = doctor.objects.filter(specialization=doctortype)


        return render(request,'patient/consult_a_doctor/consult_a_doctor.html',{"dobj":dobj , 'puser' : puser})

   


def  make_consultation(request, doctorusername):

    if request.method == 'POST':
       

        patientusername = request.session['patientusername']
        puser = User.objects.get(username=patientusername)
        patient_obj = puser.patient
        
        
        #doctorusername = request.session['doctorusername']
        duser = User.objects.get(username=doctorusername)
        doctor_obj = duser.doctor
        request.session['doctorusername'] = doctorusername


        diseaseinfo_id = request.session['diseaseinfo_id']
        diseaseinfo_obj = diseaseinfo.objects.get(id=diseaseinfo_id)

        consultation_date = date.today()
        status = "active"
        
        consultation_new = consultation( patient=patient_obj, doctor=doctor_obj, diseaseinfo=diseaseinfo_obj, consultation_date=consultation_date,status=status)
        consultation_new.save()

        request.session['consultation_id'] = consultation_new.id

        print("consultation record is saved sucessfully.............................")

         
        return redirect('consultationview',consultation_new.id)



def  consultationview(request,consultation_id):
   
    if request.method == 'GET':
      puser = None
      try:
         patientusername = request.session['patientusername']
         puser = User.objects.get(username=patientusername)
      except:
         pass
      request.session['consultation_id'] = consultation_id
      consultation_obj = consultation.objects.get(id=consultation_id)

      return render(request,'consultation/consultation.html', {"consultation":consultation_obj , 'puser' : puser})

   #  if request.method == 'POST':
   #    return render(request,'consultation/consultation.html' )





def rate_review(request,consultation_id):
   if request.method == "POST":
         
         consultation_obj = consultation.objects.get(id=consultation_id)
         patient = consultation_obj.patient
         doctor1 = consultation_obj.doctor
         rating = request.POST.get('rating')
         review = request.POST.get('review')

         rating_obj = rating_review(patient=patient,doctor=doctor1,rating=rating,review=review)
         rating_obj.save()

         rate = int(rating_obj.rating_is)
         doctor.objects.filter(pk=doctor1).update(rating=rate)
         

         return redirect('consultationview',consultation_id)





def close_consultation(request,consultation_id):
   if request.method == "POST":
         
         consultation.objects.filter(pk=consultation_id).update(status="closed")
         
         return redirect('home')






#-----------------------------chatting system ---------------------------------------------------


def post(request):
    if request.method == "POST":
        msg = request.POST.get('msgbox', None)

        consultation_id = request.session['consultation_id'] 
        consultation_obj = consultation.objects.get(id=consultation_id)

        c = Chat(consultation_id=consultation_obj,sender=request.user, message=msg)

        #msg = c.user.username+": "+msg

        if msg != '':            
            c.save()
            print("msg saved"+ msg )
            return JsonResponse({ 'msg': msg })
    else:
        return HttpResponse('Request must be POST.')



def chat_messages(request):
   if request.method == "GET":

         consultation_id = request.session['consultation_id'] 

         c = Chat.objects.filter(consultation_id=consultation_id)
         return render(request, 'consultation/chat_body.html', {'chat': c})


#-----------------------------chatting system ---------------------------------------------------


