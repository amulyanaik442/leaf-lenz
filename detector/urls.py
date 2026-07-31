from django.urls import path
from .views import PredictView, FeedbackView, HistoryView, StatsView, DiseaseListView, home

urlpatterns = [
    path('', home, name='home'),
    path('predict/', PredictView.as_view(), name='predict'),
    path('feedback/', FeedbackView.as_view(), name='feedback'),
    path('history/', HistoryView.as_view(), name='history'),
    path('stats/', StatsView.as_view(), name='stats'),
    path('diseases/', DiseaseListView.as_view(), name='diseases'),
]
