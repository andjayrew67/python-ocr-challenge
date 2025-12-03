FROM mcr.microsoft.com/azure-functions/python:4-python3.11

ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true

# Install Tesseract and Pandoc
RUN apt-get update && \
    apt-get install -y \
        tesseract-ocr \
        tesseract-ocr-eng \
        libtesseract-dev \
        libleptonica-dev \
        pandoc \
        && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /
RUN pip install --upgrade pip && pip install -r /requirements.txt

COPY . /home/site/wwwroot
