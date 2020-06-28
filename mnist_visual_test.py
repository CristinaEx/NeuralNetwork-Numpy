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

        self.writeboard_items = []
        

    def create_widgets(self):
        self.fm1=tk.Frame(self)
        self.fm1.grid(column = 0)

        self.txt1 = tk.StringVar()
        self.txt1.set('点击RandomTest进行随机测试')
        self.label1 = tk.Label(self.fm1,textvariable=self.txt1).grid(row=0)
        self.txt2 = tk.StringVar()
        self.txt2.set('当前未进行测试')
        self.label2 = tk.Label(self.fm1,textvariable=self.txt2).grid(row=2)
        self.b_random_test = tk.Button(self.fm1)
        self.b_random_test["text"] = "RandomTest"
        self.b_random_test["command"] = self.randomTest
        self.b_random_test.grid(row = 3)
        self.b_test_all = tk.Button(self.fm1)
        self.b_test_all["text"] = "TestAll(检测速率很慢)"
        self.b_test_all["command"] = self.testAll
        self.b_test_all.grid(row = 4)

        self.fm2=tk.Frame(self)
        self.fm2.grid(column = 1,row = 0)
        self.label2 = tk.Label(self.fm2,text='手写测试').grid(row=0)
        self.writeboard = tk.Canvas(self.fm2, width=200, height=200)
        self.writeboard.grid(row = 1)
        self.writeboard.bind("<B1-Motion>",self.paint)
        self.writeboard_array = numpy.zeros([200,200])

        self.fm3=tk.Frame(self.fm2)
        self.fm3.grid(row=3)
        self.b_clear = tk.Button(self.fm3)
        self.b_clear["text"] = "clear"
        self.b_clear["command"] = self.clear
        self.b_clear.grid(column=0,row = 0)
        self.b_count = tk.Button(self.fm3)
        self.b_count["text"] = "count"
        self.b_count["command"] = self.count
        self.b_count.grid(column=1,row = 0)

        self.txt3 = tk.StringVar()
        self.txt3.set('当前未进行测试')
        self.label3 = tk.Label(self.fm2,textvariable=self.txt3).grid(row=2)

    def count(self):
        data = self.writeboard_array
        # 归一化为mnist形式
        top_y=0
        down_y=len(data)
        number = False
        for i in range(len(data)):
            useful = False
            if number:
                useful = True
            for j in range(len(data[i])):
                if data[i][j] > 100:
                    if number:
                        useful = False
                        break
                    else:
                        number = not number
                        top_y = i
                    break
            if useful:
                down_y = i
                break
        top_x=0
        down_x=len(data[0])
        number = False
        for i in range(len(data[0])):
            useful = False
            if number:
                useful = True
            for j in range(len(data)):
                if data[j][i] > 100:
                    if number:
                        useful = False
                        break
                    else:
                        number = not number
                        top_x = i
                    break
            if useful:
                down_x = i
                break
        data = data[top_y:down_y,top_x:down_x]
        pad_p=0.15
        if down_y-top_y>down_x-top_x:
            pad_y = int((down_y-top_y)*pad_p)
            pad_x = int((down_y-top_y-(down_x-top_x))/2+pad_y)
        else:
            pad_x = int((down_x-top_x)*pad_p)
            pad_y = int((down_x-top_x-(down_y-top_y))/2+pad_x)
        data = numpy.pad(data,((pad_y,pad_y),(pad_x,pad_x)),'constant')
        im = Image.fromarray(data)
        im = im.convert('L')
        im = im.resize((28,28))
        data = numpy.array(im)
        data = numpy.reshape(data,[1,28,28,1])
        self.net.addData(data)
        result = self.net.count()[0,0,0,:]
        j = 0
        for i in range(len(result)):
            if result[i] > result[j]:
                j = i
        self.txt3.set("net result:"+str(j))

    def clear(self):
        self.writeboard_array = numpy.zeros([200,200])
        for item in self.writeboard_items:
            self.writeboard.delete(item)
        self.writeboard_items = []
        self.txt3.set('当前未进行测试')

    def paint(self,event):
        pad = 13
        pad_div=int(pad/2)
        x1,y1 = (event.x - pad_div), (event.y - pad_div)
        x2,y2 = (event.x + pad_div), (event.y + pad_div)
        item = self.writeboard.create_oval(x1, y1, x2, y2, fill="red")
        self.writeboard_items.append(item)
        for i in range(pad):
            for j in range(pad):
                try:
                    self.writeboard_array[event.y-i+pad_div][event.x-j+pad_div] = 255
                except:
                    pass

    def randomTest(self):
        index = random.randint(1,len(self.test_images)-2)
        data = self.test_images[index]
        im = Image.fromarray(data)
        im = im.convert('L')
        # im = im.resize((400,400))
        render= ImageTk.PhotoImage(im)  
        img = tk.Label(self.fm1,image=render)
        img.image = render
        img.grid(row=1)
        # net
        data = numpy.reshape(data,[1,28,28,1])
        label = self.test_labels[index]
        self.txt1.set('label:'+str(int(label)))
        self.net.addData(data)
        result = self.net.count()[0,0,0,:]
        j = 0
        for i in range(len(result)):
            if result[i] > result[j]:
                j = i
        self.txt2.set("net result:"+str(j))

    def testAll(self):
        correct_num = 0
        error_num = 0
        for i in range(len(self.test_images)):
            if i % 1000 == 0:
                self.txt1.set("test"+str(i*1000)+'~'+str((i+1)*1000)+'...')
            data = self.test_images[i]
            data = numpy.reshape(data,[1,28,28,1])
            label = self.test_labels[i]
            self.net.addData(data)
            result = self.net.count()[0,0,0,:]
            j = 0
            for i in range(len(result)):
                if result[i] > result[j]:
                    j = i
            if int(label) == j:
                correct_num += 1
            else:
                error_num += 1
        self.txt1.set("finish!")
        self.txt2.set("correct_rate:"+str(correct_num/(correct_num+error_num)))

if __name__ == '__main__':
    root = tk.Tk()
    # root.geometry("500x500")
    mainview = MnistVisualTest(master=root)
    mainview.mainloop()