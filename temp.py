import signal


print("press ctrl+c for interruption") 


def xyz():

    print("im xyz")

   

def keyboardInterruptHandler(signal, frame):

    xyz()

    #print("call your function here".format(signal))

    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)

while True:

    pass