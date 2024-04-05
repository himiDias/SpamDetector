import tkinter as tk

class EmailGUI:
    def __init__(self, master):
        self.master = master
        master.title("Email GUI")

        # Create a label and a text box
        self.email_label = tk.Label(master, text="Email:")
        self.email_label.pack()
        self.email_textbox = tk.Text(master, height=5, font=("Arial", 12))
        self.email_textbox.pack(pady=10 , padx = 10)

        # Create a submit button
        self.submit_button = tk.Button(master, text="Submit", command=self.save_email)
        self.submit_button.pack()

    def save_email(self):
        email = self.email_textbox.get("1.0", tk.END).strip()
        listofexpressions = ["£","won",".com","click","hot","xxx","$","!","gamble","%","security","alert"]
        inputs =[]
        for i in listofexpressions:
            x = email.lower().count(i)
            inputs.append(x)
        print (inputs)

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("800x200") # set the window size to 400x200
    email_gui = EmailGUI(root)
    print(email_gui)
    root.mainloop()




    #if __name__ == '__main__':