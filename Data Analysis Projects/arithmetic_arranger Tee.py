def arithmetic_arranger(problems, solution=False):
#output lines: (4 is for when set to true)    
    if len(problems) > 5:
        return "Error: Too many problems."
  
   #setting up problems input: 
    line1 = ""
    line2 = ""
    line3 = ""
    line4 = ""

    for i in problems:
#^^split the problem into 1st number, operator, and 2nd number 
        a = str(i).split()
        num1 = a[0]
        op = a[1]
        num2 = a[2]

        if not op in ["+","-"]:
            return "Error: Operator must be '+' or '-'."


        if not num1.isnumeric() or not num2.isnumeric():
            return "Error: Numbers must only contain digits."


        if len(num1) > 4 or len(num2) > 4:
            return "Error: Numbers cannot be more than four digits."


        if op == "+":
            solution = int(num1) + int(num2)
        else:
            solution = int(num1) - int(num2)
            
            
            #set the length for rjust:
        num_length = len(max([num1, num2],key = len))
            #^^the biggest number of num1/num2
        top = num1.rjust(num_length + 2)
            #^^first line is right-adjusted 2 char more than the len of max num
        bottom = op+num2.rjust(num_length + 1)
            #^^2nd line is operator and num2 right-adjusted 1 char more than len(max)
        line = "-"*(num_length + 2)
            #^^3rd line is the ----; len of max num +2
        result = str(solution).rjust(num_length + 2)
            #^^4th line is the answer, in string form, right-adjusted 2 char more than len of biggest number


        if i != problems[-1]:
            line1 += top + '    '
            line2 += bottom + '    '
            line3 += line + '    '
            line4 += result + '    '
        else:
            line1 += top
            line2 += bottom
            line3 += line
            line4 += result
#make the problems display vertically by adding newline:
        vertical_problems = line1+'\n'+line2+'\n'+line3

    if solution:
        vertical_problems+='\n'+line4
    return(vertical_problems)