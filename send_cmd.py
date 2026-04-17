import socket
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python send_cmd.py [start|stop|reset]")
        sys.exit(1)
        
    cmd = sys.argv[1].lower()
    if cmd not in ["start", "stop", "reset"]:
        print(f"Error: Unknown command '{cmd}'. Must be start, stop, or reset.")
        sys.exit(1)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(cmd.encode('utf-8'), ("192.168.10.46", 8888))
    print(f"Successfully sent '{cmd.upper()}' teleop_command to Isaac Lab!")

if __name__ == "__main__":
    main()
