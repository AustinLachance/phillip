import enum
import os
from threading import Thread
from . import util

@enum.unique
class Button(enum.Enum):
    A = 0
    B = 1
    X = 2
    Y = 3
    Z = 4
    START = 5
    L = 6
    R = 7
    D_UP = 8
    D_DOWN = 9
    D_LEFT = 10
    D_RIGHT = 11

# List of moves the controller can perform
class Moves(enum.Enum):
    # Single Inputs
    null = 0
    A = 1
    B = 2
    Z = 3
    Shield = 4
    Left = 5
    Right = 6
    Up = 7
    UpLeft = 8
    UpRight = 9
    UpTilt = 10
    Down = 11
    DownLeft = 12
    DownRight = 13
    DownTilt = 14
    LeftSmash = 15
    RightSmash = 16
    UpSmash = 17
    DownSmash = 18
    Jump = 19
    LeftA = 20
    RightA = 21
    UpA = 22
    DownA = 23
    UpTiltA = 24
    DownTiltA = 25
    LeftB = 26
    RightB = 27
    UpB = 28
    UpLeftB = 29
    UpRightB = 30
    DownB = 31
    LeftJump = 32
    RightJump = 33
    JumpZ = 34
    LeftJumpZ = 35
    RightJumpZ = 36
    LeftShield = 37
    RightShield = 38
    AShield = 39

# Same as above but for Benjamin Rodriquez Vars' implementation
class benMoves(enum.Enum):
    A = 0
    AShield = 1
    B = 2
    Down = 3
    DownB = 4
    DownSmash = 5
    Jump = 6
    JumpZ = 7
    Left = 8
    LeftA = 9
    LeftB = 10
    LeftJump = 11
    LeftJumpZ = 12
    LeftShield = 13
    LeftSmash = 14
    null = 15
    Right = 16
    RightA = 17
    RightB = 18
    RightJump = 19
    RightJumpZ = 20
    RightShield = 21
    RightSmash = 22
    Shield = 23
    Up = 24
    UpB = 25
    UpSmash = 26
    UpTiltA = 27
    Z = 28
# moveStrDict = {
#     "A" : "PRESS A",
#     "B": "PRESS B",
#     "Z": "PRESS Z",
#     "Shield": = "SET L 1",
#     "Left": "SET MAIN 0 .5",
#     "Right": "SET MAIN 1 .5",
#     "Up": "SET MAIN .5 1",
#     "Down": "SET MAIN .5 0",
#     "UpLeft": ""

#     }

@enum.unique
class Trigger(enum.Enum):
    L = 0
    R = 1

@enum.unique
class Stick(enum.Enum):
    MAIN = 0
    C = 1

class Pad:
    """Writes out controller inputs."""
    def __init__(self, path, tcp=False):
        """Opens the fifo. Blocks until the other end is listening.
        Args:
          path: Path to pipe file.
          tcp: Whether to use zmq over tcp or a fifo. If true, the pipe file
            is simply a text file containing the port number. The port will
            be a hash of the path.
        """
        print("PIPE PATH: " + str(path))
        self.tcp = tcp
        if tcp:
          import zmq
          context = zmq.Context()
          port = util.port(path)
          
          with open(path, 'w') as f:
            f.write(str(port))

          self.socket = context.socket(zmq.PUSH)
          address = "tcp://127.0.0.1:%d" % port
          print("Binding pad %s to address %s" % (path, address))
          self.socket.bind(address)
        else:
            print("PIPE PATH: " + str(path))
            os.mkfifo(path)
            self.pipe = open(path, 'w', buffering=1)
            print(str(self.pipe))
        
        self.message = ""

    def __del__(self):
        """Closes the fifo."""
        if not self.tcp:
            self.pipe.close()
    
    def write(self, command, buffering=False):
        # print(buffering)
        self.message += command + '\n'
        
        if not buffering:
            self.flush()
    
    def flush(self):
        if self.tcp:
            #print("sent message", self.message)
            self.socket.send_string(self.message)
        else:
            # print("sending input!!!")
            # print(self.message)
            self.pipe.write(self.message)
        self.message = ""

    def press_button(self, button, buffering=False):
        """Press a button."""
        assert button in Button
        self.write('PRESS {}'.format(button.name), buffering)

    def perform_move(self, move):
        moveName = move.name

        if 'A' == moveName:
            self.write('PRESS A')
        elif 'B' == moveName:
            self.write('PRESS B')
        elif 'Z' == moveName:
            self.write('PRESS Z')
        elif 'Up' == moveName:
            self.write('SET MAIN .5 1')
        elif 'Down' == moveName:
            self.write('SET MAIN .5 0')
        elif 'Left' == moveName:
            self.write('SET MAIN 0 .5')
        elif 'Right' == moveName:
            self.write('SET MAIN 1 .5')
        elif 'UpLeft'== moveName:
            self.write('SET MAIN 0 1')
        elif 'UpRight' == moveName:
            self.write('SET MAIN 1 1')
        elif 'DownLeft' == moveName:
            self.write('SET MAIN 0 1')
        elif 'UpTilt' == moveName:
            self.write('SET MAIN .5 .8')
        elif 'DownTilt' == moveName:
            self.write('SET MAIN .5 .2')
        elif 'LeftSmash' == moveName:
            self.write('SET C 0 .5')
        elif 'RightSmash' == moveName:
            self.write('SET C 1 .5')
        elif 'UpSmash' == moveName:
            self.write('SET C .5 1')
        elif 'DownSmash' == moveName:
            self.write('SET C .5 0')
        elif 'Jump' == moveName:
            self.write('PRESS X')
        elif 'LeftA' == moveName:
            self.write('SET MAIN 0 .5')
            self.write('PRESS A')
        elif 'RightA' == moveName:
            self.write('SET MAIN 1 .5')
            self.write('PRESS A')
        elif 'UpA' == moveName:
            self.write('SET MAIN .5 1')
            self.write('PRESS A')
        elif 'DownA' == moveName:
            self.write('SET MAIN .5 0')
            self.write('PRESS A')
        elif 'UpTiltA' == moveName:
            self.write('SET MAIN .5 .67')
            self.write('PRESS A')
        elif 'DownTiltA' == moveName:
            self.write('SET MAIN .5 .2')
            self.write('PRESS A')
        elif 'LeftB' == moveName:
            self.write('SET MAIN 0 .5')
            self.write('PRESS B')
        elif 'RightB' == moveName:
            self.write('SET MAIN 1 .5')
            self.write('PRESS B')
        elif 'UpB' == moveName:
            self.write('SET MAIN .5 1')
            self.write('PRESS B')
        elif 'DownB' == moveName:
            self.write('SET MAIN .5 0')
            self.write('PRESS B')
        elif 'UpLeftB' == moveName:
            self.write('SET MAIN 0 1')
            self.write('PRESS B')
        elif 'UpRightB' == moveName:
            self.write('SET MAIN 1 1')
            self.write('PRESS B')
        elif 'LeftJump' == moveName:
            self.write('SET MAIN 0 .5')
            self.write('PRESS X')
        elif 'RightJump' == moveName:
            self.write('SET MAIN 1 .5')
            self.write('PRESS X')
        elif 'JumpZ' == moveName:
            self.write('PRESS X')
            self.write('PRESS Z')
        elif 'LeftJumpZ' == moveName:
            self.write('SET MAIN 0 .5')
            self.write('PRESS X')
            self.write('PRESS Z')
        elif 'RightJumpZ' == moveName:
            self.write('SET MAIN 1 .5')
            self.write('PRESS X')
            self.write('PRESS Z')
        elif 'Shield' == moveName:
            self.write('SET L 1')
        elif 'LeftShield' == moveName:
            self.write('SET MAIN 0 .5')
            self.write('SET L 1')
        elif 'RightShield' == moveName:
            self.write('SET MAIN 1 .5')
            self.write('SET L 1')
        elif 'AShield' == moveName:
            self.write('PRESS A')
            self.write('SET L 1')

    def release_button(self, button, buffering=False):
        """Release a button."""
        assert button in Button
        self.write('RELEASE {}'.format(button.name), buffering)

    def press_trigger(self, trigger, amount, buffering=False):
        """Press a trigger. Amount is in [0, 1], with 0 as released."""
        assert trigger in Trigger
        # assert 0 <= amount <= 1
        self.write('SET {} {:.2f}'.format(trigger.name, amount), buffering)

    def tilt_stick(self, stick, x, y, buffering=False):
        """Tilt a stick. x and y are in [0, 1], with 0.5 as neutral."""
        assert stick in Stick
        try:
          assert 0 <= x <= 1 and 0 <= y <= 1
        except AssertionError:
          import ipdb; ipdb.set_trace()
        self.write('SET {} {:.2f} {:.2f}'.format(stick.name, x, y), buffering)

    def send_controller(self, controller):
        for button in Button:
            field = 'button_' + button.name
            if hasattr(controller, field):
                if getattr(controller, field):
                    self.press_button(button, True)
                else:
                    self.release_button(button, True)

        # for trigger in Trigger:
        #     field = 'trigger_' + trigger.name
        #     self.press_trigger(trigger, getattr(controller, field))

        for stick in Stick:
            field = 'stick_' + stick.name
            value = getattr(controller, field)
            self.tilt_stick(stick, value.x, value.y, True)
        
        self.flush()
