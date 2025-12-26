"""
Django settings for urban_climate project.
"""

from pathlib import Path
import os
import platform
import environ

BASE_DIR = Path(__file__).resolve().parent.parent

# Initialize environment variables
env = environ.Env(
    DEBUG=(bool, False)
)
environ.Env.read_env(os.path.join(BASE_DIR, '.env'))

# ----- Platform detection & GDAL/GEOS selection -----
def _is_wsl() -> bool:
    try:
        # WSL kernels include 'Microsoft' in /proc/version
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except Exception:
        return False

IS_WINDOWS = (platform.system() == 'Windows')
IS_WSL = _is_wsl() or (platform.system() == 'Linux' and 'WSL' in os.environ.get('WSL_DISTRO_NAME', ''))

# Prefer explicit GDAL/GEOS env, fall back to OS-specific env names, then sensible defaults
if IS_WINDOWS and not IS_WSL:
    GDAL_LIBRARY_PATH = os.environ.get('GDAL_LIBRARY_PATH') or os.environ.get('GDAL_LIBRARY_PATH_WIN')
    GEOS_LIBRARY_PATH = os.environ.get('GEOS_LIBRARY_PATH') or os.environ.get('GEOS_LIBRARY_PATH_WIN')
else:
    # Linux/WSL
    GDAL_LIBRARY_PATH = (
        os.environ.get('GDAL_LIBRARY_PATH')
        or os.environ.get('GDAL_LIBRARY_PATH_WSL')
        or '/usr/lib/x86_64-linux-gnu/libgdal.so'
    )
    GEOS_LIBRARY_PATH = (
        os.environ.get('GEOS_LIBRARY_PATH')
        or os.environ.get('GEOS_LIBRARY_PATH_WSL')
        or '/usr/lib/x86_64-linux-gnu/libgeos_c.so'
    )

# Debugging hints: print resolved GIS lib paths and DB connection info when DEBUG
try:
    if os.environ.get('DEBUG_GIS', '1') == '1':
        print(f"[settings] Platform: Windows={IS_WINDOWS} WSL={IS_WSL}")
        print("[settings] GDAL_LIBRARY_PATH:", GDAL_LIBRARY_PATH or '(not set)')
        print("[settings] GEOS_LIBRARY_PATH:", GEOS_LIBRARY_PATH or '(not set)')
        if GDAL_LIBRARY_PATH and not os.path.exists(GDAL_LIBRARY_PATH):
            print(f"[settings][warn] GDAL path does not exist: {GDAL_LIBRARY_PATH}")
        if GEOS_LIBRARY_PATH and not os.path.exists(GEOS_LIBRARY_PATH):
            print(f"[settings][warn] GEOS path does not exist: {GEOS_LIBRARY_PATH}")
except Exception:
    # Never fail settings import due to debug printing
    pass

# ----- Database host/port resolution (support Windows host from WSL) -----
DB_NAME_ENV = env('DB_NAME', default='urban_climate')
DB_USER_ENV = env('DB_USER', default='postgres')
DB_PASSWORD_ENV = env('DB_PASSWORD', default='postgres')
# Respect explicit DB_HOST; fallback to sensible default if unset
DB_HOST_ENV = env('DB_HOST', default=None) or '10.255.255.254'
if IS_WSL:
    DB_PORT_ENV = env('DB_PORT', default='5432')
else:
    DB_PORT_ENV = env('DB_PORT', default='5433')

def _wsl_windows_host_ip() -> str | None:
    # Try env override first
    ip = os.environ.get('WSL_WINDOWS_HOST')
    if ip:
        return ip
    # Fallback: use nameserver from resolv.conf (usually Windows host IP)
    try:
        with open('/etc/resolv.conf', 'r') as f:
            for line in f:
                if line.strip().startswith('nameserver'):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        return parts[1]
    except Exception:
        pass
    return None

WSL_USE_WINDOWS_HOST = env.bool('WSL_USE_WINDOWS_HOST', default=False)

# Only rewrite localhost to Windows host IP when explicitly requested
if IS_WSL and WSL_USE_WINDOWS_HOST and DB_HOST_ENV in ('localhost', '127.0.0.1'):
    host_candidate = _wsl_windows_host_ip()
    if host_candidate:
        DB_HOST_ENV = host_candidate

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = env('SECRET_KEY', default='django-insecure-change-this-in-production')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = env('DEBUG', default=True)

ALLOWED_HOSTS = ['localhost', '127.0.0.1', '0.0.0.0']

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.gis',
    'rest_framework',
    'corsheaders',
    'core',
    'api',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'urban_climate.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'urban_climate.wsgi.application'

# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases
DATABASES = {
    'default': {
        'ENGINE': 'django.contrib.gis.db.backends.postgis',
        'NAME': DB_NAME_ENV,
        'USER': DB_USER_ENV,
        'PASSWORD': DB_PASSWORD_ENV,
        'HOST': DB_HOST_ENV,
        'PORT': DB_PORT_ENV,
    }
}

# Optional: log DB host/port in debug to help diagnose port 5432 vs 5433 differences
try:
    if os.environ.get('DEBUG_DB', '1') == '1':
        print(f"[settings] DB -> host={DATABASES['default']['HOST']} port={DATABASES['default']['PORT']}")
        if IS_WSL:
            print("[settings] WSL detected; if connecting to Windows PostgreSQL, host is", DATABASES['default']['HOST'])
except Exception:
    pass

# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/
STATIC_URL = 'static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# Media files
MEDIA_URL = 'media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Django REST Framework
REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 100,
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
    ],
}

# CORS settings
CORS_ALLOWED_ORIGINS = [
    "http://localhost:5173",  # Vite default port
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

CORS_ALLOW_CREDENTIALS = True

# Celery Configuration
CELERY_BROKER_URL = env('CELERY_BROKER_URL', default='redis://localhost:6379/0')
CELERY_RESULT_BACKEND = env('CELERY_RESULT_BACKEND', default='redis://localhost:6379/0')
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = TIME_ZONE

# Cache configuration (using Redis)
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': env('REDIS_URL', default='redis://localhost:6379/1'),
    }
}

