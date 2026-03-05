import sys
from typing import Any

def error_message_detail(error:Any):
    _,_,exc_tb = sys.exc_info()

    if exc_tb is None:
        return f"Error message [{str(error)}]"
    
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message

class CustomException(Exception):
    
    def __init__(self, error_message:Any):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message)

    def __str__(self):
        return self.error_message 