import argparse

class ArgError(Exception):
    def __init__ (self, message, parser):
        self.message = message
        self.parser:argparse.ArgumentParser = parser
        self.parser.print_usage()
        self.parser.exit()
