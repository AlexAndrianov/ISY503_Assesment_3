from django.db import models
#from nlp_model.bert_model import analyse_text

# Create your models here.

class EstimatingItem(models.Model):
    analysed_text = models.CharField(max_length=400) 
    analysis_result = models.IntegerField(null=False, default=0) # -1: negative, 0: non defined, 1 positive

    def analyse(self, input_text):
        #self.analysis_result =  analyse_text(input_text)
        return self.analysis_result