set FLOWER_UNAUTHENTICATED_API=true
python -m celery -A celery_task flower -b redis://127.0.0.1  --loglevel=INFO