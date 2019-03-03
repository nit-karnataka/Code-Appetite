from django.conf import settings
from django.db import models
from django.utils import timezone


class Record(models.Model):
    Name_of_FileUploader = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    Name_of_Patient = models.CharField(max_length=200)
    FieldOne = models.TextField()
    FieldTwo = models.TextField()
    Date_of_Prescription = models.DateTimeField(default=timezone.now)
    Date_of_Uploading = models.DateTimeField(blank=True, null=True)

    def publish(self):
        self.Date_of_Uploading  = timezone.now()
        self.save()

    def __str__(self):
        return self.Name_of_Patient
