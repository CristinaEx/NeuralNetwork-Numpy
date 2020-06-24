import tkinter as tk
from PIL import Image,ImageTk
from mnist_load import *
from net import Net
from mnist_train import MODEL_PATH
import random
import numpy

class MnistVisualTest(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()
        self.test_images = load_test_images()
        self.test_labels = load_test_labels()
        self.net = Net()
        self.net.load(MODEL_PATH)

    def create_widgets(self):
        self.b_random_test = tk.Button(self)
        self.b_random_test["text"] = "RandomTest"
        self.b_random_test["command"] = self.randomTest
        self.b_random_test.grid(row = 3)

    def randomTest(self):
        index = random.randint(1,len(self.test_images)-2)
        data = self.test_images[index]
        im = Image.fromarray(data)
        im = im.convert('L')
        # im = im.resize((400,400))
        render= ImageTk.PhotoImage(im)  
        img = tk.Label(self,image=render)
        img.image = render
        img.grid(row=1)
        # net
        data = numpy.reshape(data,[1,28,28,1])
        label = self.test_labels[index]
        tk.Label(self,text="label:"+str(int(label))).grid(row=0)
        tk.Label(self,text="counting...").grid(row=2)
        self.net.addData(data)
        result = self.net.count()[0,0,0,:]
        j = 0
        for i in range(len(result)):
            if result[i] > result[j]:
                j = i
        tk.Label(self,text="net result:"+str(j)).grid(row=2)

if __name__ == '__main__':
    root = tk.Tk()
    # root.geometry("500x500")
    mainview = MnistVisualTest(master=root)
    mainview.mainloop()