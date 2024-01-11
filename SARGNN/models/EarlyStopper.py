#!/usr/bin/python
class EarlyStopper:

    def stop(self, epoch, val_loss, val_acc=None, test_loss=None, test_acc=None, train_loss=None, train_acc=None):
        raise NotImplementedError("Implement this method!")

    def get_best_vl_metrics(self):

        return self.train_loss, self.train_acc,self.val_loss,self.val_acc,self.best_epoch



class Patience(EarlyStopper):

    def __init__(self, patience=20, use_loss=True):
        self.local_val_optimum = float("inf") if use_loss else -float("inf")
        self.use_loss = use_loss
        self.patience = patience
        self.best_epoch = -1
        self.counter = -1

        self.train_loss, self.train_acc = None, None
        self.val_loss, self.val_acc = None, None

    def stop(self, epoch, val_loss, val_acc=None, train_loss=None, train_acc=None):
        if self.use_loss:
            if val_loss <= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = val_loss
                self.best_epoch = epoch
                self.train_loss, self.train_acc = train_loss, train_acc
                self.val_loss, self.val_acc = val_loss, val_acc
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience
        else:
            if val_acc >= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = val_acc
                self.best_epoch = epoch
                self.train_loss, self.train_acc = train_loss, train_acc
                self.val_loss, self.val_acc = val_loss, val_acc
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience



