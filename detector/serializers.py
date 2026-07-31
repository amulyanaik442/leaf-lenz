from rest_framework import serializers
from .models import PredictionLog
from .disease_info import DISEASE_INFO

class PredictionLogSerializer(serializers.ModelSerializer):
    description = serializers.SerializerMethodField()
    symptoms = serializers.SerializerMethodField()
    treatment_prevention = serializers.SerializerMethodField()

    class Meta:
        model = PredictionLog
        fields = [
            'id', 'image', 'predicted_label', 'confidence', 'plant_name', 
            'disease_name', 'created_at', 'is_correct', 'user_feedback',
            'corrected_label', 'description', 'symptoms', 'treatment_prevention'
        ]
        read_only_fields = ['id', 'predicted_label', 'confidence', 'plant_name', 'disease_name', 'created_at']

    def get_description(self, obj):
        details = DISEASE_INFO.get(obj.predicted_label, DISEASE_INFO.get("fallback", {}))
        return details.get("description", "N/A")

    def get_symptoms(self, obj):
        details = DISEASE_INFO.get(obj.predicted_label, DISEASE_INFO.get("fallback", {}))
        return details.get("symptoms", "N/A")

    def get_treatment_prevention(self, obj):
        details = DISEASE_INFO.get(obj.predicted_label, DISEASE_INFO.get("fallback", {}))
        return details.get("treatment_prevention", "N/A")

