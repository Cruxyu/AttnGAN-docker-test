# AttnGAN using docker and streamlit

![example](https://downloader.disk.yandex.ru/preview/fea3a1fb42366ace7f726b422644b94f4cca7096a031542ffde8e1a9e068f9c8/6095bf9f/w84pwgl-ht2Vw4ZyORpgVm7xfVVVI2_0t12E7gDhkzISYSnjjnWa-sXm6RNIkGG9Xs1KvrbfpF0X8wH1JJx23g%3D%3D?uid=0&filename=example.png&disposition=inline&hash=&limit=0&content_type=image%2Fpng&owner_uid=0&tknv=v2&size=2048x2048)

## EN

python3.8 with translate module all inside docker container and streamlit lib.

Requirements:

- Docker
- 4GB or more RAM
- 20 GB or less free space 

To try AttnGAN borrowed from [original authors](https://github.com/taoxugit/AttnGAN):

- Download this repo or simply `git clone https://github.com/Cruxyu/AttnGAN-docker-test`
- Download pre-trained models [link](https://disk.yandex.ru/d/QEZO4mNo2pvTEw?w=1) and extract all inside AttnGAN folder
- Run inside main folder `docker build -t AttnGAN-test -f dockerfile .`
- Then `docker run -p 8051:8051 AttnGAN-test`
- Now you can open inside browser `localhost:8051`

## RU

python3.8 с переводчиком внутри контейнера docker с streamlit библиотекой

Требования

- Docker
- 4 ГБ или больше ОЗУ
- 20 ГБ или меньше свободного места

Чтобы попробовать AttnGAN позаимственной у [авторов оригинала](https://github.com/taoxugit/AttnGAN):

- Скачайте этот репозиторий или просто `git clone https://github.com/Cruxyu/AttnGAN-docker-test`
- Скачайте предобученные модели по [ссылке](https://disk.yandex.ru/d/QEZO4mNo2pvTEw?w=1) и распакуйте их в основную папку
- Далее в командной строке внутри этой папки `docker build -t AttnGAN-test -f dockerfile .` 
- И затем запускаем  `docker run -p 8051:8051 AttnGAN-test`
- Теперь вы можете открыть программу в браузере `localhost:8051`
