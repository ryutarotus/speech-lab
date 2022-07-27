from django.db import models

# Create your models here.
class Upload(models.Model):
    document = models.FileField(upload_to='media', default='settings.MEDIA_ROOT/VOICEACTRESS100_012.wav')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    text = models.CharField(max_length=52)