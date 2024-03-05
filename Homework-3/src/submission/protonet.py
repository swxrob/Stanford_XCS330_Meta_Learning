"""Implementation of prototypical networks for Omniglot."""
import sys
sys.path.append('..')
import argparse
import os

import numpy as np
import torch

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torch import nn
import torch.nn.functional as F  # pylint: disable=unused-import
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch.utils import tensorboard

import omniglot
import util  # pylint: disable=unused-import

NUM_INPUT_CHANNELS = 1
NUM_HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
PRINT_INTERVAL = 10
VAL_INTERVAL = PRINT_INTERVAL * 5
NUM_TEST_TASKS = 600

'''
RJR: Section for assisting with Debugging
'''
import os
if 'IS_GRADESCOPE_ENV' not in os.environ:
    PRINT_DEBUG = False
else:
    PRINT_DEBUG = False


class ProtoNetNetwork(nn.Module):
    """Container for ProtoNet weights and image-to-latent computation."""

    def __init__(self, device):
        """Inits ProtoNetNetwork.

        The network consists of four convolutional blocks, each comprising a
        convolution layer, a batch normalization layer, ReLU activation, and 2x2
        max pooling for downsampling. There is an additional flattening
        operation at the end.

        Note that unlike conventional use, batch normalization is always done
        with batch statistics, regardless of whether we are training or
        evaluating. This technically makes meta-learning transductive, as
        opposed to inductive.

        Args:
            device (str): device to be used
        """
        super().__init__()
        layers = []
        in_channels = NUM_INPUT_CHANNELS
        for _ in range(NUM_CONV_LAYERS):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    NUM_HIDDEN_CHANNELS,
                    (KERNEL_SIZE, KERNEL_SIZE),
                    padding='same'
                )
            )
            layers.append(nn.BatchNorm2d(NUM_HIDDEN_CHANNELS))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = NUM_HIDDEN_CHANNELS
        layers.append(nn.Flatten())
        self._layers = nn.Sequential(*layers)
        self.to(device)

    def forward(self, images):
        """Computes the latent representation of a batch of images.

        Args:
            images (Tensor): batch of Omniglot images
                shape (num_images, channels, height, width)

        Returns:
            a Tensor containing a batch of latent representations
                shape (num_images, latents)
        """
        return self._layers(images)


class ProtoNet:
    """Trains and assesses a prototypical network."""

    def __init__(self, learning_rate, log_dir, device, compile=False, backend=None):
        """Inits ProtoNet.

        Args:
            learning_rate (float): learning rate for the Adam optimizer
            log_dir (str): path to logging directory
            device (str): device to be used
        """
        self.device = device
        self._network = ProtoNetNetwork(device)

        if(compile == True):
            try:
                self._network = torch.compile(self._network, backend=backend)
                print(f"ProtoNetNetwork model compiled")
            except Exception as err:
                print(f"Model compile not supported: {err}")

        self._optimizer = torch.optim.Adam(
            self._network.parameters(),
            lr=learning_rate
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0

    def _step(self, task_batch):
        """Computes ProtoNet mean loss (and accuracy) on a batch of tasks.

        Args:
            task_batch (tuple[Tensor, Tensor, Tensor, Tensor]):
                batch of tasks from an Omniglot DataLoader

        Returns:
            a Tensor containing mean ProtoNet loss over the batch
                shape ()
            mean support set accuracy over the batch as a float
            mean query set accuracy over the batch as a float
        """
        loss_batch = []
        accuracy_support_batch = []
        accuracy_query_batch = []
        for task in task_batch:
            images_support, labels_support, images_query, labels_query = task
            images_support = images_support.to(self.device)
            labels_support = labels_support.to(self.device)
            images_query = images_query.to(self.device)
            labels_query = labels_query.to(self.device)

            ### START CODE HERE ###
            my_name = 'ProtoNet:_step()'
            if PRINT_DEBUG: print(my_name+': STARTING...')

            '''
            STEPS Overall:
            
            Implement Handout equation (5) Loss, J(theta): 

            Equation 5:  J(theta) = 1/B * sum_over_B( 1/NQ * sum_over_Dts( Cross Entropy Loss ) )
            Equation 3:  CEL = basically -log likelihood of softmax( logits )
            Equation 2:  Logits = Distance between samples and class prototypes
            Equation 1:  Class prototypes = mean of K feature vectors of this classes support images 
            '''

            '''
            STEP 0: Setup - Useful dimensions and notation
            
            Extract dimensions for convenience:
                N (way): number of classes in a task
                K (shot): number of support examples per class
                Q: number of query examples per class
                B: number of tasks in a batch (=> Outside of this Loop)
                I: 2D I*I dimension

            Supporting Images and Labels appear to be laid out as:
            Support:
                labels_support = [K labels for Class 1, ... , K labels for Class N]    
                    Test 1b-0-basic: vector as: [ 5] as in [0,1,2,3,4]
                    Handout 1.b: 5-way, 5-shot: [25=N(5)*K(5)] as in [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4]
                images_support = [K images for Class 1, ... , K labels for Class N]    
                    Test 1b-0-basic: matrix as: [5(N=5, K=1),1,28,28]
                    Handout 1.b: 5-way, 5-shot: [25:N(5)*K(5),1,28,28]
            Query:
                labels_query = [Q labels for Class 1, ..., Q labels for Class N]
                    Test 1b-0-basic: vector as: 
                    Handout 1.b: 5-way, 5-shot: [75=N(5)*Q(15)] as in [15x0's, 15x1's,..., 15x4's]
                images_query = [Q images for Class 1, ..., Q images for Class N]
                    Handout 1.b: 5-way, 5-shot: [75=N(5)*Q(15),1,28,28] as in [15 images for Class 1, ..., 15 images for Class N]
            '''
            N = torch.unique(labels_support).shape[0]
            K = int(labels_support.shape[0] / N)
            Q = int(labels_query.shape[0] / N)
            I = images_support.shape[2]

            '''
            STEP 1: Compute Class Prototypes
            
            Equation 1: cn =  mean of the K feature vectors of that class’s support images            
            '''
            # Prototypes (cn) as Support Image feature representations via our CNN
            # Don't track Gradients for these Model Evaluation steps
            with torch.no_grad():

                # Support Image embeddings shaped as [K*N, latent dimension]
                # FIXME: We can't cast to Double (float64) when using the Apple MPS framework due to this error:
                #        "TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead."
                #        "-[IOGPUMetalCommandBuffer validate]:215: failed assertion `commit an already committed command buffer'"
                #        We wanted to: Cast to double (float64) to ensure we are doing precise math going forward.
                #        Also note: Our CNN self._network() doesn't accept Double (float64) as input, so we cast the output embeddings.
                images_support_embed = self._network(images_support)

                # FIXME: Shouldn't this be dim=0? (but CF said it should be dim=1)
                # Setting this to dim=0:
                #     FAILS for: python protonet.py --num_support 5
                #     SUCCESSES: python protonet.py --num_support 1
                #                AND is required for passing 1b-0-basic and 1b-1-hidden
                # Setting this to dim=1:
                #      FAILS for: python protonet.py --num_support 1
                #      SUCCESSES: python protonet.py --num_support 5
                # => Obviously something strange about the reshaping.
                # cn = images_support_embed.reshape(N, K, -1).mean(dim=1)
                # From the CF:

                # FIXME: As described above, something about this code only works for:
                #            for the N-way = 5, K-support = 5 case when dim=1,
                #            for the N-way = 5, K-support = 1 case when dim=0,
                #            for 1b-0-basic and 1b-1-hidden Autograder when dim=0
                #        So, for now, we will heavily constrain when this code can be used until
                #        we have a more general robust solution.
                assert N==5 and (K==1 or K==5), my_name+": Assertion Failure, this code isn't general enough for other than N=5 and K=1 or K=5."
                if K == 1:
                    cn = images_support_embed.reshape(K, N, -1).mean(dim=0)
                elif K == 5:
                    cn = images_support_embed.reshape(K, N, -1).mean(dim=1)

                '''
                STEP 2: Compute a measure of distance d between fθ(x) and each of the prototypes. 
                
                We will use the squared Euclidean distance, Equation 2:
                    d(fθ(x), cn) = ∥fθ(x) − cn∥_2^2
                We can use the PyTorch p-2 norm function torch.cdist(x1, x2, p=2)
                    x1: fθ(x), our query images
                    x2: cn, our prototype features
                '''
                # Euclidean distance between Class prototypes and Support Images
                d_squared = torch.cdist(cn, images_support_embed, p=2) ** 2
                # Manual way, using broadcasting. Good hint here: https://discuss.pytorch.org/t/how-to-calculate-pair-wise-differences-between-two-tensors-in-a-vectorized-way/37451
                # FIXME: d_squared = ((cn.unsqueeze(1) - images_support_embed) ** 2).sum(axis=1).sqrt() ** 2

                # Support Label Logits and Predicted Labels
                #     Hint, Logits.argmax and Predicted Labels should be nearly == to true Support labels
                #     Will be a little off equality from the Centroid when we have more than 1 Support Image per Class
                labels_support_logits = -d_squared.T
                # For Debugging here are our Predicted Support Labels:
                labels_support_pred = labels_support_logits.argmax(axis=1)

                # Accuracy Support Image / Labels:
                #     This will always be = 100% in the case of 1 Support Image / Class (task)
                #     and <100% when # Support Images > 1 (i.e. slightly different than the Prototype centroid)
                # Function util.score( logits, labels )
                #     logits [Samples (K), Classes (N)]: Unormalized values predicted by model
                #     labels [Samples (K) x Classes (N) (i.e. total Samples)]: classification labels from 0 to num_classes - 1
                accuracy_support = util.score(labels_support_logits.detach(), labels_support)
                if PRINT_DEBUG: print(my_name+': Accuracy Support Labels:', accuracy_support)


            '''
            STEP 3: Minimize the negative log likelihood of the Query data.
            
            Note: Cross Entropy Loss == Negative Log Likelihood( Softmax (logits) )
            Note: We already trained the model to generalize by computing prototypes (cn) using Support Data. 
            '''
            # Obtain Query Image embeddings in latent feature space using our CNN
            images_query_embed = self._network(images_query)

            # Distance between our Query Images and Prototypes in the latent / embedding space
            d_squared = torch.cdist(cn, images_query_embed, p=2) ** 2
            # Manual way, using broadcasting. Good hint here: https://discuss.pytorch.org/t/how-to-calculate-pair-wise-differences-between-two-tensors-in-a-vectorized-way/37451
            # FIXME: d_squared = ((cn.unsqueeze(1) - images_query_embed) ** 2).sum(axis=1).sqrt() ** 2

            '''
            STEP 3a: Logits
            We interpret the negative squared distances as logits, or unnormalized log-probabilities, of x belonging to each class.
            '''
            labels_query_logits = -d_squared.T
            # For Debugging here are our Predicted Query Labels:
            labels_query_pred = labels_query_logits.argmax(axis=1)

            # Accuracy for the Query Image Labels:
            accuracy_query = util.score(labels_query_logits.detach(), labels_query)

            # For Debugging:
            # Normalize logits to real probabilities s.t. the sum of probabilities of a Query Sample over N classes == 1.
            # prob_x_class = torch.softmax(labels_query_logits.reshape((Q, N, N)), dim=2).reshape((Q*N,N))
            # Double check our probabilities are normalized per class (i.e. sum to 1)
            # assert torch.allclose(prob_x_class.sum(axis=1), torch.ones((Q*N))), my_name+': Error in normalizing probabilities using Softmax.'


            '''
            STEP 3b: Minimize the negative log likelihood of the Query data using Cross Entropy on the Logits.
            '''
            loss = F.cross_entropy(input=labels_query_logits, target=labels_query)

            '''
            STEP 4: Save the task minibatches loss, accuracy_support, accuracy_query
            '''
            loss_batch.append( loss )
            accuracy_support_batch.append( accuracy_support )
            accuracy_query_batch.append( accuracy_query )

            if PRINT_DEBUG:
                print(my_name+': DONE this inner loop, with loss:', loss, ', accuracy_support: ', accuracy_support,
                      ', accuracy_query: ', accuracy_query)
            ### END CODE HERE ###
        return (
            torch.mean(torch.stack(loss_batch)),
            np.mean(accuracy_support_batch),
            np.mean(accuracy_query_batch)
        )

    def train(self, dataloader_train, dataloader_val, writer, args):
        """Train the ProtoNet.

        Consumes dataloader_train to optimize weights of ProtoNetNetwork
        while periodically validating on dataloader_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_train (DataLoader): loader for train tasks
            dataloader_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
            args (dict): dictionary with all parameters
        """
        print(f'Starting training at iteration {self._start_train_step}.')
        for i_step, task_batch in enumerate(
                dataloader_train,
                start=self._start_train_step
        ):
            self._optimizer.zero_grad()
            loss, accuracy_support, accuracy_query = self._step(task_batch)
            loss.backward()
            self._optimizer.step()

            if i_step % PRINT_INTERVAL == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {loss.item():.3f}, '
                    f'support accuracy: {accuracy_support.item():.3f}, '
                    f'query accuracy: {accuracy_query.item():.3f}'
                )
                writer.add_scalar('loss/train', loss.item(), i_step)
                writer.add_scalar(
                    'train_accuracy/support',
                    accuracy_support.item(),
                    i_step
                )
                writer.add_scalar(
                    'train_accuracy/query',
                    accuracy_query.item(),
                    i_step
                )

            if i_step % VAL_INTERVAL == 0:
                with torch.no_grad():
                    losses, accuracies_support, accuracies_query = [], [], []
                    for val_task_batch in dataloader_val:
                        loss, accuracy_support, accuracy_query = (
                            self._step(val_task_batch)
                        )
                        losses.append(loss.item())
                        accuracies_support.append(accuracy_support)
                        accuracies_query.append(accuracy_query)
                    val_loss = np.mean(losses)
                    val_accuracy_support = np.mean(accuracies_support)
                    val_accuracy_query = np.mean(accuracies_query)
                print(
                    f'Validation: '
                    f'loss: {val_loss:.3f}, '
                    f'support accuracy: {val_accuracy_support:.3f}, '
                    f'query accuracy: {val_accuracy_query:.3f}'
                )
                writer.add_scalar('loss/val', val_loss, i_step)
                writer.add_scalar(
                    'val_accuracy/support',
                    val_accuracy_support,
                    i_step
                )
                writer.add_scalar(
                    'val_accuracy/query',
                    val_accuracy_query,
                    i_step
                )

            if i_step % SAVE_INTERVAL == 0:
                self._save(i_step)

                with open(f'protonet_results_{args.num_support}_{args.num_way}.npy', 'wb') as f:
                    np.save(f, val_loss)
                    np.save(f, val_accuracy_support)
                    np.save(f, val_accuracy_query)

    def test(self, dataloader_test):
        """Evaluate the ProtoNet on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        accuracies = []
        for task_batch in dataloader_test:
            accuracies.append(self._step(task_batch)[2])
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )

    def load(self, checkpoint_step, filename=""):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load
            filename (str): directly setting name of checkpoint file, default ="", when argument is passed, then checkpoint will be ignored

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{checkpoint_step}.pt'
        ) if filename == "" else filename
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._network.load_state_dict(state['network_state_dict'])
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            self._start_train_step = checkpoint_step + 1
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def _save(self, checkpoint_step):
        """Saves network and optimizer state_dicts as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        torch.save(
            dict(network_state_dict=self._network.state_dict(),
                 optimizer_state_dict=self._optimizer.state_dict()),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')


def main(args):

    print(args)

    if args.device == "gpu" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        DEVICE = "mps"
    elif args.device == "gpu" and torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    print("Using device: ", DEVICE)

    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/protonet/omniglot.way_{args.num_way}.support_{args.num_support}.query_{args.num_query}.lr_{args.learning_rate}.batch_size_{args.batch_size}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    protonet = ProtoNet(args.learning_rate, log_dir, DEVICE, args.compile, args.backend)

    if args.checkpoint_step > -1:
        protonet.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        num_training_tasks = args.batch_size * (args.num_train_iterations -
                                                args.checkpoint_step - 1)
        print(
            f'Training on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_train = omniglot.get_omniglot_dataloader(
            'train',
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            num_training_tasks,
            args.num_workers
        )
        dataloader_val = omniglot.get_omniglot_dataloader(
            'val',
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            args.batch_size * 4,
            args.num_workers,
        )
        protonet.train(
            dataloader_train,
            dataloader_val,
            writer,
            args
        )
    else:
        print(
            f'Testing on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_test = omniglot.get_omniglot_dataloader(
            'test',
            1,
            args.num_way,
            args.num_support,
            args.num_query,
            NUM_TEST_TASKS,
            args.num_workers
        )
        protonet.test(dataloader_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a ProtoNet!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--num_way', type=int, default=5,
                        help='number of classes in a task')
    parser.add_argument('--num_support', type=int, default=1,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=15,
                        help='number of query examples per class in a task')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for the network')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=5000,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--num_workers', type=int, default=2, 
                        help=('needed to specify omniglot dataloader'))
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction)
    parser.add_argument("--backend", type=str, default="inductor", choices=['inductor', 'aot_eager', 'cudagraphs'])
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    if args.cache == True:
        # Download Omniglot Dataset
        if not os.path.isdir("./omniglot_resized"):
            gdd.download_file_from_google_drive(
                file_id="1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI",
                dest_path="./omniglot_resized.zip",
                unzip=True,
            )
        assert os.path.isdir("./omniglot_resized")
    else:
        main(args)
