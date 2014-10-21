import re

class TaggedDocument: 
    def __init__(self,data): 
        self.data = String
        # dictionary containing tags and offsets for those tags.
        self.offsets = {}
        
    def tags(self):
        return d.keys()
            
    def regexp(self): 
        """Recursively applies regex to a string, tagging at start and stop sites"""
        
