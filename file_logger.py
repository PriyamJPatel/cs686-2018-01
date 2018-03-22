class file_logger: 

    """
    Constructor
    """
    def __init__(self, log_level, fname = "file_log.txt"):
        self.__log_level__ = log_level
        self.file = open(fname,"a")

    def log(self, log_level, message):
        if(self.__log_level__ >= log_level):
            self.file.write(str(log_level) + ":" + message + "\n")


