from django import forms

class Text_Estimate_Form(forms.Form):
    input_text = forms.CharField( 
        widget=forms.Textarea(attrs={'rows': 4, 'cols': 40})
    )

    CHOICES = [
        (1, 'Naive_Bayes_Model_With_Simple_Tokenizer'),
        (2, 'Stacking_Classifier_Logistic_Regression_Plus_SVC'),
        (3, 'PyTorch_Neural_Network_Model'),
    ]
    nlp_model = forms.ChoiceField(choices=CHOICES, widget=forms.Select)