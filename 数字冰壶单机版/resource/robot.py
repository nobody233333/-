# -*- coding: utf-8 -*-
import socket 
import time
import argparse

#python 与客户端连接

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-p','--port', help='tcp server port', default="7788", required=False)
parser.add_argument('-host','--host', help='host', default="127.0.0.1", required=False)
args, unknown = parser.parse_known_args()

#连接host(无需修改)
host=args.host
#默认连接端口(无需修改)
port=int(args.port)

obj=socket.socket()
obj.connect((host,port))

def send_message(sock, message):
    message_with_delimiter = message
    sock.sendall(message_with_delimiter.encode())

def recv_message(sock):
    buffer = bytearray()
    while True:
        data = sock.recv(1)
        if not data or data == b'\0':
            break
        buffer.extend(data)
    return buffer.decode()

#初始化
shotnum=str("0")
order=str("Player1")#先后手
state=[]

#策略
def strategy(state_list,order):
    bestshot=str("BESTSHOT 6 2.2 0")
    return bestshot
    
retNullTime = 0
while True:
    ret = recv_message(obj)
    messageList = ret.split(" ")
    if ret == "":
        retNullTime = retNullTime + 1
    if retNullTime == 5:
        break
    if messageList[0] == "NAME":
        order=messageList[1]
    if messageList[0]=="ISREADY":
        time.sleep(0.5)
        send_message(obj, "READYOK")
        time.sleep(0.5)
        send_message(obj, "NAME Robot")
    if messageList[0]=="POSITION":
        if state:
            state=[]
        state.append(ret.split(" ")[1:31])
    if messageList[0]=="SETSTATE":
        shotnum=ret.split(" ")[1]        
        state.append(shotnum)
    if messageList[0]=="GO":
        shot=strategy(state,order)
        send_message(obj, shot)
    if messageList[0]=="MOTIONINFO":
        x_coordinate = float(messageList[1])
        y_coordinate = float(messageList[2])
        x_velocity = float(messageList[3])
        y_velocity = float(messageList[4])
        angular_velocity = float(messageList[5])
        