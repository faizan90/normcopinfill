
import io
import sys
#import importlib
#
#importlib.reload(sys)
#sys.setdefaultencoding('utf8')

__all__ = ['StdFileLogger', 'StdFileLoggerCtrl']


class StdFileLogger:
    '''To write stdout to screen and a file

    Got it from stackoverflow
    '''

    def __init__(self, filename):
        '''Initiate stuff
        '''
        self.terminal = sys.stdout
        self.logfile = io.open(filename, "a", encoding='utf-8')
        return

    def write(self, message):
        '''Write messages to screen and file
        '''
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()
        return

    def flush(self):
        '''Needed for python3'''
        return


class StdFileLoggerCtrl:
    '''Make StdFileLogger easier to use

    Also from stackoverflow
    '''
    def __init__(self, filename):
        """Start transcript, appending print output to given filename"""
        sys.stdout = StdFileLogger(filename)
        return

    def stop(self):
        """Stop transcript and return print functionality to normal"""
        sys.stdout.logfile.close()
        sys.stdout = sys.stdout.terminal
        return
