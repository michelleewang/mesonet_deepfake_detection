import numpy as np
import matplotlib.pyplot as plt
import time
from classifiers import *
from pipeline import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def showInstructions():

    instructions = input("Would you like to see the instructions? (y/n): ")

    while True:
        if instructions == "y":
            print("\n------------------------------------------------------")
            print("\nHow to play: ")
            print("1. Enter the number of rounds you would like to play.")
            print("2. For each round, we will show you an image. Exit out of the image to guess.")
            print("3. The AI will also make predictions as to whether the image is a deepfake or a real image.")
            print("4. For each image that you guess correctly, you get a point! Try to beat the AI.")
            print("\n------------------------------------------------------")
            print("\nLet's begin!")
            return

        elif instructions == "n":
            print("\n------------------------------------------------------")
            print("\nLet's begin!")
            return

        else:
            instructions = input("Please enter y/n: ")

def askRounds():
    rounds = input("How many rounds would you like to play? (pick between 1-10): ")

    while True:
        if rounds.isdigit() and int(rounds) <= 10:
            #print("%i rounds." % int(rounds))
            return int(rounds)
        else:
            rounds = input("Please enter at least 1 round: ")

def playRound(classifier, generator):

    userPoints = 0
    computerPoints = 0
    # render image X with label y for MesoNet
    X, y = generator.next()
    # show image
    plt.imshow(np.squeeze(X))
    plt.show()

    # prompt user for guess
    userInput = input("Do you think this image is real or a deepfake? (enter 'real' or 'deepfake'): ")
    while True:
        if userInput == "real":
            userPred = 1
            break
        elif userInput == "deepfake":
            userPred = 0
            break
        else:
            userInput = input("Please enter 'real' or 'deepfake'): ")

    modelPred = np.round(classifier.predict(X))     # make model prediction
    realClass = y[0]                                # real class

    # show result
    if realClass == 0:
        print("\nThe image was a deepfake!")
    else:
        print("\nThe image was real!")

    # Evaluating user prediction
    if userPred == realClass:
        print("You were correct! 1 point for the user.")
        userPoints += 1
    else:
        print("Sorry, you were incorrect! 0 points for the user.")

    # evaluating model prediction
    if modelPred == realClass:
        print("The computer was correct! 1 point for the computer.")
        computerPoints += 1
    else:
        print("The computer was incorrect! 0 points for the computer.")

    return userPoints, computerPoints

def main():

    userPoints = 0
    computerPoints = 0

    # Load the model and its pretrained weights
    classifier = Meso4()
    classifier.load('weights/Meso4_DF.h5')

    # Minimial image generator
    dataGenerator = ImageDataGenerator(rescale=1./255)
    generator = dataGenerator.flow_from_directory(
            'train_test',
            target_size=(256, 256),
            batch_size=1,
            class_mode='binary',
            #subset='training'
            )
    print("\n------------------------------------------------------")
    print("\nWelcome to the deepfake quiz! With this quiz, you will compete against an AI to guess whether or not images are real or deepfakes.\n")

    showInstructions()

    rounds = askRounds()

    for i in range(rounds):
        print("\n------------------------------------------------------")
        print("\nRound %i!" % (int(i)+1))
        userPointsNew, computerPointsNew = playRound(classifier, generator)
        userPoints += userPointsNew
        computerPoints += computerPointsNew

        print("User points: %i " % userPoints)
        print("Computer points: %i " % computerPoints)

    print("\n------------------------------------------------------")
    if userPoints > computerPoints:
        print("\nYou win with %i points. Great job!" % userPoints)
    elif userPoints < computerPoints:
        print("\nThe computer wins with %i points. Better luck next time." % computerPoints)
    else:
        print("\nIt was a tie with %i points. Good job!" % userPoints)

main()
