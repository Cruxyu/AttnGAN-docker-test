FROM pytorch/pytorch:latest

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY environment.yml /usr/src/app/
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "dipl", "/bin/bash", "-c"]
RUN pip install --no-deps easynmt \
&& pip install tqdm transformers numpy nltk sentencepiece  

COPY . /usr/src/app

EXPOSE 8501
EXPOSE 8080

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "dipl", "streamlit","run", "--server.address", "0.0.0.0", "--server.port","8080", "main.py"]

