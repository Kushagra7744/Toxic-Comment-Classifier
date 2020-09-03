import socket
import sys
import time
import select


client=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
if len(sys.argv)!=3:
	print("script_name IP_address port_no")
	exit()

host_name=sys.argv[1]
port=int(sys.argv[2])

client.connect((host_name,port))
print("Connection established")

while(1):
	socket_list= [sys.stdin,client]#possible input streams

	read_sockets, write_sockets, error_sockets=select.select(socket_list,[],[])

	for socket in read_sockets:
		if socket==client:
			msg= socket.recv(2048)
			msg.decode()
			print(msg)
			print('\n');
		else:
			msg= sys.stdin.readline()
			msg=msg.encode()
			client.send(msg)
			sys.stdout.write("<YOU>")
			sys.stdout.write(msg.decode())
			sys.stdout.flush()

client.close()
