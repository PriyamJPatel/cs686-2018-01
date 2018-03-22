class stdout_logger:

    """
    Constructor
    """
    def __init__(self, log_level):
        self.__log_level__ = log_level

    def log(self, log_level, message):
        if(self.__log_level__ >= log_level):
            print(str(log_level) + ":" + message)

