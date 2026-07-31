from django.shortcuts import render
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from .models import PredictionLog
from .serializers import PredictionLogSerializer
from .inference import predict_leaf_disease, predict_with_wheat_fallback, predict_by_crop
from .disease_info import DISEASE_INFO

def parse_class_label(label):
    """
    Parses a class label like 'Tomato___Bacterial_spot' into plant and disease names.
    """
    if "___" in label:
        parts = label.split("___")
        plant = parts[0].replace("_", " ").title()
        disease = parts[1].replace("_", " ").title()
        # Handle 'healthy' naming explicitly
        if disease.lower() == "healthy":
            disease = "Healthy"
    else:
        plant = "Unknown Plant"
        disease = label.replace("_", " ").title()
    
    return plant, disease

class PredictView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response(
                {"error": "No image file provided. Key must be 'image'."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        image_file = request.FILES['image']
        crop = request.data.get('crop', 'auto')
        print(f"[VIEW DEBUG] crop='{crop}' type={type(crop)} keys={list(request.data.keys())}")
        
        # Save log entry first to generate image path
        log_entry = PredictionLog(image=image_file)
        
        # Run prediction on the file
        try:
            predicted_label, confidence, top5, crop_mismatch = predict_by_crop(image_file, crop)
        except Exception as e:
            return Response(
                {"error": f"ML Inference failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        # Parse plant and disease details and enforce a confidence threshold
        CONFIDENCE_THRESHOLD = 0.15
        print(f"[DEBUG] Raw prediction: '{predicted_label}' | Confidence: {confidence:.4f} ({confidence*100:.1f}%) | crop={crop}")
        if confidence < CONFIDENCE_THRESHOLD and crop == 'auto':
            predicted_label = "no_leaf"
            plant_name = "No Leaf Detected"
            disease_name = "Invalid Image"
        else:
            plant_name, disease_name = parse_class_label(predicted_label)
        
        # Update log details and save to database
        log_entry.predicted_label = predicted_label
        log_entry.confidence = confidence
        log_entry.plant_name = plant_name
        log_entry.disease_name = disease_name
        log_entry.save()
        
        # Fetch disease details from disease_data.json
        details = DISEASE_INFO.get(predicted_label)
        if not details:
            # Fallback to general details
            details = DISEASE_INFO.get("fallback", {
                "plant_name": plant_name,
                "disease_name": disease_name,
                "description": "No specific details available in local database for this condition.",
                "symptoms": "N/A",
                "treatment_prevention": "N/A"
            })
            
        return Response({
            "status": "success",
            "prediction_id": log_entry.id,
            "image_url": request.build_absolute_uri(log_entry.image.url),
            "prediction": {
                "plant": plant_name,
                "disease": disease_name,
                "confidence": confidence,
                "raw_label": predicted_label
            },
            "disease_info": {
                "description": details.get("description"),
                "symptoms": details.get("symptoms"),
                "treatment_and_prevention": details.get("treatment_prevention")
            },
            "crop_mismatch": crop_mismatch
        }, status=status.HTTP_200_OK)

class FeedbackView(APIView):
    def post(self, request, *args, **kwargs):
        prediction_id = request.data.get('prediction_id')
        is_correct = request.data.get('is_correct')
        user_feedback = request.data.get('user_feedback', '')
        corrected_label = request.data.get('corrected_label', None)

        if prediction_id is None or is_correct is None:
            return Response(
                {"error": "Both 'prediction_id' and 'is_correct' are required fields."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            log_entry = PredictionLog.objects.get(id=prediction_id)
            log_entry.is_correct = bool(is_correct)
            if user_feedback:
                log_entry.user_feedback = user_feedback
            if corrected_label is not None:
                log_entry.corrected_label = corrected_label
            log_entry.save()
            return Response({"status": "feedback saved successfully"}, status=status.HTTP_200_OK)
        except PredictionLog.DoesNotExist:
            return Response(
                {"error": f"Prediction with ID {prediction_id} not found."},
                status=status.HTTP_404_NOT_FOUND
            )

class HistoryView(APIView):
    def get(self, request, *args, **kwargs):
        # Return newest logs first
        logs = PredictionLog.objects.all().order_by('-created_at')[:50]
        serializer = PredictionLogSerializer(logs, many=True, context={'request': request})
        return Response(serializer.data, status=status.HTTP_200_OK)

class StatsView(APIView):
    def get(self, request, *args, **kwargs):
        from django.db.models import Count
        total_scans = PredictionLog.objects.count()
        correct_scans = PredictionLog.objects.filter(is_correct=True).count()
        incorrect_scans = PredictionLog.objects.filter(is_correct=False).count()
        reviewed_scans = correct_scans + incorrect_scans
        
        accuracy_rate = round(correct_scans / reviewed_scans, 4) if reviewed_scans > 0 else 1.0
        
        # Get plant distribution
        plant_distribution_query = (
            PredictionLog.objects.values('plant_name')
            .annotate(count=Count('id'))
            .order_by('-count')
        )
        plant_distribution = list(plant_distribution_query)
        
        return Response({
            "total_scans": total_scans,
            "correct_scans": correct_scans,
            "incorrect_scans": incorrect_scans,
            "accuracy_rate": accuracy_rate,
            "plant_distribution": plant_distribution
        }, status=status.HTTP_200_OK)

class DiseaseListView(APIView):
    def get(self, request, *args, **kwargs):
        return Response(DISEASE_INFO, status=status.HTTP_200_OK)

def home(request):
    return render(request, 'detector/index.html')
