import multiprocessing
import threading
import time
from llama import chat_loop
from yolov10 import webcam
# Create a pipe for communication between the conversation loop and object detection loop
conversation_pipe, object_detection_pipe = multiprocessing.Pipe()

# Create a thread for the object detection loop
#object_detection_thread = threading.Thread(target=webcam, args=(object_detection_pipe,))
# Create a thread for the conversation loop
#conversation_thread = threading.Thread(target=chat_loop, args=(conversation_pipe,))


# Create a thread for the object detection loop
object_detection_thread = threading.Thread(target=webcam, args=(object_detection_pipe,))
object_detection_thread.start()
print(object_detection_pipe.recv)
# Create a thread for the conversation loop
conversation_thread = threading.Thread(target=chat_loop)
conversation_thread.start()


conversation_thread.join()
object_detection_thread.join()


#received_data = object_detection_pipe.recv()
#conversation_pipe.send(processed_data.encode())

