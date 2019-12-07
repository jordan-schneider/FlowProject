import random
import string
import sys
def randomString(stringLen = 128):
    choices = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(choices) for _ in range(stringLen))

if __name__ == '__main__':
    filename = 'password.txt'
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
    with open(filename, 'w') as f:
        f.write(randomString())
