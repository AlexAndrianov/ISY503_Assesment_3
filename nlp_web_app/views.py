from django.shortcuts import render, HttpResponse
from .forms import Text_Estimate_Form
from .models import EstimatingItem

# Create your views here.

def NLP_view(request):
    result = ""
    if request.method == 'POST':
        form = Text_Estimate_Form(request.POST)
        if form.is_valid():
            input_text = form.cleaned_data['input_text']

            # Create a new estimator instance
            estimator = EstimatingItem.objects.create()
            
            # Analyze the input text using the TextProcessor instance
            analysis_res = estimator.analyse(input_text)

            if analysis_res == 1:
                result = "The sentiment is positive"
            elif analysis_res == -1:
                result = "The sentiment is negative"
            else:
                result = "The sentiment is not defined"
    else:
        form = Text_Estimate_Form()

    return render(request, 'index.html', {'form': form, 'result': result})
