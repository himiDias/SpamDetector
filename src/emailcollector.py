#YOU HAVE WON £1000 AS THE LUCKY WINNER OF THE TESCO WEEKLY CUSTOMER LOTTERY, CLICK THE LINK BELOW TO CLAIM YOUR PRIZE"

end = False
while end == False:
    email = input("ENTER EMAIL CONTENTS: ")
    listofexpressions = ["won","£","link","click","hot","sex"]

    inputs =[]

    for i in listofexpressions:
        x = email.lower().count(i)
        inputs.append(x)


    print(inputs)
    ennnd = input("END?")
    if ennnd =="yes":
        end = True