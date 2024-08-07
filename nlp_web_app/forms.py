from django import forms

class Text_Estimate_Form(forms.Form):
    input_text = forms.CharField( 
        widget=forms.Textarea(attrs={'rows': 4, 'cols': 40})
    )