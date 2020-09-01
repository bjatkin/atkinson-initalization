import pandas as pd
import matplotlib.pyplot as plt

class Results():
    def __init__(self):
        self.losses = []
        self.validation_losses = []
        self.validation_losses_index = []
        self.accuracies = []
        self.accuracies_index = []
    
    def clear(self):
        self.accuracies = []
        self.losses = []
        self.validation_losses = []

    def add_loss(self, loss):
        self.losses.append(loss)
    
    def add_validation_loss(self, index, loss):
        self.validation_losses_index.append(index)
        self.validation_losses.append(loss)
    
    def add_accuracy(self, index, accuracy):
        self.accuracies_index.append(index)
        self.accuracies.append(accuracy)
    
    def show_loss_plot(self, title="training losses"):
        plt.plot(self.losses, label="training loss")
        plt.plot(self.validation_losses_index, self.validation_losses, label="validation loss")
        plt.xlabel("Batches")
        plt.ylabel("Loss Score")
        plt.legend()
        plt.title(title)
        plt.show()

    def show_accuracy_plot(self, title="training accuracy"):
        plt.plot(self.accuracies_index, self.accuracies)
        plt.xlabel("Batches")
        plt.ylabel("Accruacy")
        plt.legend()
        plt.title(title)
        plt.show()