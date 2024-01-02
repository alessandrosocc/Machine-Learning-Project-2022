from libraries import *

def clear_terminal ():
    #posix è il nome dell'os per linux o mac
    if(os.name == 'posix'):
        os.system('clear')
    # else lo schermo verrà pulito per un os windows
    else:
        os.system('cls')

