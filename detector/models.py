from django.db import models

class PredictionLog(models.Model):
    image = models.ImageField(upload_to='leaf_uploads/')
    predicted_label = models.CharField(max_length=255)
    confidence = models.FloatField()
    plant_name = models.CharField(max_length=100)
    disease_name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    is_correct = models.BooleanField(null=True, blank=True)
    user_feedback = models.TextField(null=True, blank=True)
    corrected_label = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return f"{self.plant_name} - {self.disease_name} ({self.confidence:.2%})"
