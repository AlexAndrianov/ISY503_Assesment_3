from django.db import models
from .text_estimator import analyze_text

# Create your models here.

class EstimatingItem(models.Model):
    analysed_text = models.CharField(max_length=400) 
    analysis_result = models.IntegerField(null=False, default=0) # -1: negative, 0: non defined, 1 positive

    def analyse(self, input_text, model_id):
        self.analysis_result =  analyze_text(input_text, model_id)
        return self.analysis_result