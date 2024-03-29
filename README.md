# insilico_sklearn_boston

Краткое резюме: Воскресенье, по не зависящим от меня причинам было почти полностью утрачено, в итоге получился очень игрушечный пример.
Результат оценки сильно зависит от того, во что играли с точки зрения "жюри".


1. Стек технологий: flask-restful, postgresql, flask_sqlalchemy, numpy, skikit-learn, docker, docker-compose.
1. Обучение модели: скрипт learn.py лежит здесь: ml_models/learn.py 
     * Для обучения модели, ввиду специфики данных, мне показалось целесообразным использовать метод k-ближайших соседей.
     * Датасет разбивается на тестовый и обучающий. Размер тестового - 10% от объема данных.
     * Также в функцию разделения можно передать свой random seed и получить детерминированный результат.
     * learn.py перебирает количество соседей которые стоит учитывать ( от 1, до размера обучающей выборки). 
        Результаты сравниваются по метрике MSE (показалось, что в этой модели будет лучше, чем MAE & r^2 score).
        Сейчас я понимаю, что с таким датасетом использовал неправильный метод: наверное лучше было ставить порог
        не на количество соседей, а на радиус в котором они учитываются.
      * learn генерирует 2 файла: 
        -  best_model.pckl - pickle-сериализованная модель. Прочитал, что в мире skikit-learn так часто делают. Я помню, что pickle-файлы небезопасны, поэтому реализовал md5-проверку. 
                                       ( из-за недостатка времени я не вынес модель в отдельный Docker volume, не реализовал версионирование модели и её горячую замену, поэтому проверка - бесполезна,
                                        т.к. md5 лежит в конфиге рядом)
        -  best_model_info.yaml - там лежат все метрики по модели (MAE, MSE, r^2 score), md5-hash и количество использованных соседей.


1. Что не доделано:
    *  Я не использовал boilerplate код, а flask-restful использовал в качестве эксперимента. ( до этого для api использовал:
aiohttp, pure flask или pyeve). Поэтому - организация не идеальна. Также не реализованы ошибки в виде api-ответов.
И вообще сервер не расчитан на получение неверных запросов.
    *  Нет документации.
    *  Много вещей зазардкожено - надо бы нормальный конфиг.
    *  Версионирование модели и горячая замена.
    *  Не уверен, что хорошая идея - складывать модель в Docker-контейнер. ( мне кажется - логичнее использовать volume)
    *  Я почти не занимался devops-частью, поэтому висит просто flask-сервер, без uwsgi, nginx.
    *  Нет тестов - слишком простой пример, для себя я тестировал с помощью пары строк из python-консоли. Ещё есть curl-скрипты.
1. Сервер лежит в папке flask_app по команде docker-compose up - поднимается. Она же записана в скрипте для запуска start_server.sh.

