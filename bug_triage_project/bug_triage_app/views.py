from django.shortcuts import render,redirect
from .forms import UserCreform,BugFeatureForm
from .models import User,BugFeature
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
# from bug_triage_project import settings
# Create your views here.

model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords and punctuation, and lowercase the words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    return stemmed_tokens, lemmatized_tokens
def predict_text(product, component, summary, model, vectorizer):
    # Combine the product, component, and summary into a single text
    text = ' '.join([product, component, summary])
    
    # Preprocess the text
    stemmed_tokens, lemmatized_tokens = preprocess_text(text)
    
    # Join the preprocessed tokens into a single string
    preprocessed_text = ' '.join(stemmed_tokens)  # You can choose stemmed or lemmatized tokens here
    
    # Transform the preprocessed text using the vectorizer
    features = vectorizer.transform([preprocessed_text])
    
    # Make a prediction using the trained model
    prediction = model.predict(features)
    
    return prediction
def home(request):
    return render(request,'html/index.html')
def register(request):
    if request.method == "POST":
        f = UserCreform(request.POST)
        if f.is_valid():
            f.save()
            return redirect('/lg')
    else:
        f = UserCreform()
    return render(request,'html/register.html',{'g':f})

def report_features(request):
    if request.method == 'POST':
        form = BugFeatureForm(request.POST)
        if form.is_valid():
            # Save the reported features
            product = form.cleaned_data['product']
            component = form.cleaned_data['component']
            summary = form.cleaned_data['summary']

            # Make predictions using the pre-trained model and vectorizer
            prediction = predict_text(product, component, summary, model, vectorizer)
            print(prediction)
            form.save()
            context = {'prediction': prediction}
            return render(request, 'html/result_feature.html', context)
    else:
        form = BugFeatureForm()
    return render(request, 'html/report_feature.html', {'form': form})

