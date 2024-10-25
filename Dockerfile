FROM public.ecr.aws/lambda/python:3.10.2024.10.07.10-x86_64
#FROM python:3.10
# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     python3-dev \
#     default-libmysqlclient-dev

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
COPY main.py .
COPY llm_evaluation.py .
COPY llm_recall_precision.py .
# COPY setup_nltk.py .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY .env .
RUN pip install python-dotenv

# Run NLTK setup script
# RUN python setup_nltk.py

# RUN [ "python", "-c", "import nltk; nltk.download('punkt', download_dir='/usr/local/nltk_data')" ]
RUN [ "python3", "-c", "import nltk; nltk.download('punkt')" ]
RUN [ "python3", "-c", "import nltk; nltk.download('wordnet')" ]
RUN cp -r /root/nltk_data /usr/local/share/nltk_data


#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["main.handler"]