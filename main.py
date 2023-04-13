from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from Model import *
window = Tk()
window.geometry('700x500')
window.title('Penguins')
label1 = Label(window,
              text = "Enter number of hidden layer :  ",
              font = ("Times New Roman", 10))
label1.grid(row=2,column=0)
HiddenLayerNumber=Entry(window)
HiddenLayerNumber.grid(row=3,column=0)


label2 = Label(window,
              text = "Enter number of neurons in each hidden layer :  ",
              font = ("Times New Roman", 10))
label2.grid(row=4,column=0)
number_of_neurons=Entry(window)
number_of_neurons.grid(row=5,column=0)


label3 = Label(window,
              text = "Enter learning rate :",
              font = ("Times New Roman", 10))
label3.grid(row=6,column=0)
learning_rate=Entry(window)
learning_rate.grid(row=7,column=0)


label4 = Label(window,
              text = "Enter number of epochs  :",
              font = ("Times New Roman", 10))
label4.grid(row=8,column=0)
epochs=Entry(window)
epochs.grid(row=9,column=0)
epochs.focus_set()

lbl = Label(window,
              text = "Select activation function:  ",
              font = ("Times New Roman", 10))
lbl.grid(row=10,column=0)

activationFunction = StringVar()
choosen2 = ttk.Combobox(window, width=27, textvariable=activationFunction)

# Adding combobox drop down list
choosen2['values'] = ('sigmoid',
                          'tanh',
                     )

choosen2.grid(row=11,column=0)
choosen2.current()

Bias = IntVar()
Checkbutton(window, text="Bias ", variable=Bias, onvalue=1, offvalue=0).grid(row=13, column=0)
trainButton= Button(window, text='Train', command=lambda: [train()])
trainButton.grid(row=14,column=0)
def train():

        if epochs.get()==''or learning_rate.get()=='' or activationFunction.get()=='' or number_of_neurons.get()=='' or HiddenLayerNumber.get()=='':
           msg = messagebox.showinfo("Error", "Enter  Missing Information ")
        else:
             global x_train, minMaxDF, bias, Weights, activation_fun, hidden_layers
             numberOfNeurons=number_of_neurons.get().split(',')
             counter=0
             for i in numberOfNeurons:
                 numberOfNeurons[counter]=int(i)
                 counter=counter+1
             x_train, minMaxDF, bias, Weights, activation_fun, hidden_layers=main(int(HiddenLayerNumber.get()),numberOfNeurons,int(epochs.get()),float(learning_rate.get()),activationFunction.get(),
                  Bias.get())
             # print('Hidden Layer Number  ', int(HiddenLayerNumber.get()))
             # print('Neuron Number  ', numberOfNeurons)
             # print('LearnRate  ', float(learning_rate.get()))
             # print('epoches  ',int(epochs.get()))
             # print('Activation Funvtion  ',activationFunction.get())
             # print("************************************************************")


l1 = Label(window,
              text = "Enter bill length : ",
              font = ("Times New Roman", 10))
l1.grid(row=3,column=7)

bill_length_mm=Entry(window)
bill_length_mm.grid(row=3,column=8)


l2 = Label(window,
              text = "Enter bill depth : ",
              font = ("Times New Roman", 10))

l2.grid(row=5,column=7)

bill_depth_mm=Entry(window)
bill_depth_mm.grid(row=5,column=8)


l3 = Label(window,
              text = "Enter flipper length : ",
              font = ("Times New Roman", 10))
l3.grid(row=7,column=7)

flipper_length_mm=Entry(window)
flipper_length_mm.grid(row=7,column=8)
flipper_length_mm.focus_set()

l4 = Label(window,
              text = "Enter Gender : ",
              font = ("Times New Roman", 10))
l4.grid(row=9,column=7)

Gender=Entry(window)
Gender.grid(row=9,column=8)


l5 = Label(window,
              text = "Enter body mass : ",
              font = ("Times New Roman", 10))
l5.grid(row=11,column=7)

body_mass_g=Entry(window)
body_mass_g.grid(row=11,column=8)

ButtonPredict= Button(window, text='Predict', command=lambda:[predict()])
ButtonPredict.grid(row=14,column=8)
def predict():

  if bill_length_mm.get()==''or  bill_depth_mm.get()==''or flipper_length_mm.get()==''or  Gender.get()=='' or  body_mass_g.get()=='':
      msg = messagebox.showinfo("Error", "Enter  Missing Information ")
  else:
      sampleData=[[float(bill_length_mm.get()),float(bill_depth_mm.get()),int(flipper_length_mm.get()),Gender.get(),int(body_mass_g.get())]]
      predictSample(sampleData,x_train, minMaxDF, bias, Weights, activation_fun, hidden_layers)



window.mainloop()