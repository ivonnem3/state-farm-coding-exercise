FROM python:3.11.5
WORKDIR /app


# Copy the application code
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set Port
EXPOSE 1313

CMD python ./api.py
