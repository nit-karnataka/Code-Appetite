from django.contrib import admin
from django.contrib.auth.models import Group
from .models import Record


admin.site.site_header = 'Welcome to MedRecords'
admin.site.site_title = 'MedRecords'
admin.site.index_title = 'MedRecords Administration'

admin.site.register(Record)
admin.site.unregister(Group)
