class node_str:
    node = ""
    children = list()
    def __init__(self, nodeToAdd):
        self.node = nodeToAdd

    def appendChild(self, childToAdd):
        self.children.append(childToAdd)

    def __childExit(self, child):
        if child in self.children:
            return False
        return True