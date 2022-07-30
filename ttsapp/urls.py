from django.contrib import admin
from django.urls import path
from .views import demo_func, home_func, author_func, multi_func#, my_customized_server_error

urlpatterns = [
    #path('admin/', admin.site.urls),
    path('demo/', demo_func, name="demo"),
    path('multi/', multi_func, name="multi"),
    path('home/', home_func, name="home"),
    path('author/', author_func, name="author"),
]

#handler500 = my_customized_server_error