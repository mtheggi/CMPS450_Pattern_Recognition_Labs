import numpy as np

# TODO [1]: implement the guessing_game function
def guessing_game(max : int , *, attempts: int ) -> tuple[bool, list[int] , int]: # hint: return type is tuple[bool, list[int], int]:
    random_number = np.random.randint(low =1 , high= max )
    answered : bool = False 
    Guess : list[int] = [] 
    print("You have " + str(attempts) + " attempts to guess the number.", flush=True)
    while answered == False and attempts > 0 : 
        try : 
            number = int(input("Guess a number betweem 1  and "+ str(max) + " : ")) 
                     
        except ValueError:
            raise ValueError("value must be numeric ")
            # print("Invalid input, please input integer !! " , flush=True)
            return False, [], 0
        else: 
            Guess.append(number )
            if number == random_number: 
                print("Correct" , flush=True) 
                answered = True 
                attempts -= 1
            elif number < random_number:
                print("Too low" , flush=True)
                attempts -= 1
                print(attempts , " attemps left .", flush=True) 
            else:
                print("Too high" , flush=True)
                attempts -= 1
                print(attempts , " attemps left ." , flush=True )
            
    if answered == False : 
        print("You have run out of attempts" , flush=True)

    return answered, Guess, random_number
    






# TODO [2]: implement the play_game function
def play_game() -> None:
    max_value:int = 20
    attempts:int = 5
    answered , Guesses, chosen_int= guessing_game(max_value, attempts = attempts)
    if answered == True:
        assert Guesses[-1] == chosen_int  , " Error ! , the chosen_int should be in the list " 
    else : 
        try : 
            assert chosen_int not in Guesses  
        except AssertionError:
            print("Error ! , the chosen_int should not be in the list ", flush=True) 
            x = input("Do you want to play again ? (yes/no) : ")
            if x.lower() == "yes":
                play_game()

    return
                
