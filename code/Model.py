#UR CODE HERE
import torch
import os, time
import numpy as np
from Network import MyNetwork,init_whitening_conv
from ImageUtils import parse_record

"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, configs):
        self.configs = configs
        self.network = MyNetwork(configs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.network.to(torch.device(self.device))
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=0.2, reduction='none')
        self.optimizer = None

    def model_setup(self):
        pass

    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):
        # Select optimizer
        self.optimizer = torch.optim.SGD(self.network.parameters(), 
                                         lr=configs["learning_rate"],
                                         momentum=0.9,
                                         weight_decay=configs["weight_decay"], nesterov=True)
        #self.optimizer = torch.optim.AdamW(self.network.parameters(), 
        #                                 lr=configs["learning_rate"],
        #                                 weight_decay=configs["weight_decay"])

        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // configs["batch_size"]
        init_whitening_conv(self.network.network[0], torch.tensor(x_train.reshape(x_train.shape[0], 3, 32, 32)), eps=5e-4)
        print('### Training... ###')
        
        for epoch in range(1, configs["max_epoch"]+1):
            self.network.train()
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            # Set the learning rate for this epoch
            if epoch%30 == 0:
                self.optimizer.param_groups[0]['lr'] /= 10
            
            for i in range(num_batches):
                # Construct the current batch.
                batch_size = configs["batch_size"]
                x = curr_x_train[i*batch_size : min((i+1)*batch_size, curr_x_train.shape[0]), :]
                y = curr_y_train[i*batch_size : min((i+1)*batch_size, curr_y_train.shape[0])]

                x = np.array(list(map(lambda _x: parse_record(_x,True), x)))
                x_tensor = torch.from_numpy(x).to(torch.device(self.device), dtype=torch.float)

                preds = self.network.forward(x_tensor)

                y_tensor = torch.from_numpy(y).to(torch.device(self.device)) 
                loss = self.loss(preds, y_tensor).sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))

            if epoch>100 and epoch%20==0:
                self.save(epoch)
                self.evaluate(x_valid, y_valid, True)

    def evaluate(self, x, y, valid=False):
        if not valid:
            self.load()
        self.network.eval()
        print('### Validating ###')
        preds, outputs = [],[]
        for i in range(x.shape[0]):
            with torch.no_grad():
                _x = np.expand_dims(parse_record(x[[i]], False), axis=0)
                _x = torch.from_numpy(_x).to(torch.device(self.device)) 
                self.network.to(torch.device(self.device))
                output = self.network.forward(_x.to(torch.device(self.device))) 
                pred = torch.argmax(output, dim=1)
                outputs.append(output.squeeze())
            preds.append(pred)
            ### END CODE HERE

        y = torch.tensor(y).to(torch.device(self.device))
        preds = torch.tensor(preds).to(torch.device(self.device))
        outputs = torch.stack(outputs)
        loss_valid = self.loss(outputs, y).sum()
        accu_valid = torch.sum(preds==y)/y.shape[0]
        print('Valid accuracy: {:.4f} Loss {:.6f}'.format(accu_valid, loss_valid))

    def predict_prob(self, x):
        print('### Testing ###')
        self.load()
        self.network.eval()
        preds = []
        m = torch.nn.Softmax()
        for i in range(x.shape[0]):
            with torch.no_grad():
                _x = np.expand_dims(parse_record(x[[i]], False), axis=0)
                _x = torch.from_numpy(_x).to(torch.device(self.device)).float()
                self.network.to(torch.device(self.device))
                output = self.network.forward(_x.to(torch.device(self.device))) 
            preds.append(m(output).squeeze())
        return preds
    
    def save(self, epoch):
        checkpoint_path = os.path.join(self.configs["save_dir"],
                                       'model_%s-%d.ckpt'%(self.configs["name"], epoch))
        os.makedirs(self.configs["save_dir"], exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self):
        checkpoint_name = os.path.join(self.configs["save_dir"], 'model_%s-%d.ckpt'%(self.configs["name"], 200))
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))

### END CODE HERE

