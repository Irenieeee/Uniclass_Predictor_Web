FROM python:3.12-slim

# Set working directory
WORKDIR /app

#install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

#copy requirements for better caching
COPY requirements.txt .

#Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Copy all application files
COPY . .

#Hugging face spaces uses port 7860
EXPOSE 7860

#Set environment variables
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

#run with gunicorn
CMD ["gunicorn", "app:app", "--bind","0.0.0.0:7860", "--workers","1","--timeout","300","--worker-class", "sync", "--max-requests", "1000"]