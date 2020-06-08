import re

class ObjReader:
    def __init__(self):
        self.vertexes = []
        self.faces = []

    def read(self, objpath):
        objfile = open(objpath)
        lines = objfile.readlines()
        for line in lines:
            line = re.findall(".*?(?=\n)", line)[0]
            label = re.split(" ", line)[0]
            if label == "v":
                vertex = re.split(" ", line)[1:]
                vertex = [float(x) for x in vertex]
                self.vertexes.append(vertex)
            elif label == "f":
                face = re.findall(" ([0-9]*)/", line)
                face = [int(vertex)-1 for vertex in face]
                self.faces.append(face)
    
def main():
    path = "first v1.obj"
    reader = ObjReader(path)
    reader.read()
    pass

if __name__ == "__main__":
    main()