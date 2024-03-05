#
# 5-Way, 5-Support
#

# Epoch 0

(XCS330) robr@Roberts-Laptop submission % python protonet.py --test --num_way 5 --num_support 5 --checkpoint_step 0 --log_dir logs/protonet_successfull_way_5.support_1_and_5_query/omniglot.way_5.support_1.query_15.lr_0.001.batch_size_16/ 
Namespace(log_dir='logs/protonet_successfull_way_5.support_1_and_5_query/omniglot.way_5.support_1.query_15.lr_0.001.batch_size_16/', num_way=5, num_support=5, num_query=15, learning_rate=0.001, batch_size=16, num_train_iterations=5000, test=True, checkpoint_step=0, num_workers=2, compile=None, backend='inductor', cache=False, device='cpu')
Using device:  cpu
log_dir: logs/protonet_successfull_way_5.support_1_and_5_query/omniglot.way_5.support_1.query_15.lr_0.001.batch_size_16/
Loaded checkpoint iteration 0.
Testing on tasks with composition num_way=5, num_support=5, num_query=15
Accuracy over 600 test tasks: mean 0.777, 95% confidence interval 0.006


# Epoch 1000

(XCS330) robr@Roberts-Laptop submission % python protonet.py --test --num_way 5 --num_support 5 --checkpoint_step 1000 --log_dir logs/protonet_successfull_way_5.support_1_and_5_query/omniglot.way_5.support_1.query_15.lr_0.001.batch_size_16/
Namespace(log_dir='logs/protonet_successfull_way_5.support_1_and_5_query/omniglot.way_5.support_1.query_15.lr_0.001.batch_size_16/', num_way=5, num_support=5, num_query=15, learning_rate=0.001, batch_size=16, num_train_iterations=5000, test=True, checkpoint_step=1000, num_workers=2, compile=None, backend='inductor', cache=False, device='cpu')
Using device:  cpu
log_dir: logs/protonet_successfull_way_5.support_1_and_5_query/omniglot.way_5.support_1.query_15.lr_0.001.batch_size_16/
Loaded checkpoint iteration 1000.
Testing on tasks with composition num_way=5, num_support=5, num_query=15
Accuracy over 600 test tasks: mean 0.987, 95% confidence interval 0.002


# Epoch 4900

(XCS330) robr@Roberts-Laptop submission % python protonet.py --test --num_way 5 --num_support 5 --checkpoint_step 4900 --log_dir logs/protonet_successfull_way_5.support_1_and_5_query/omniglot.way_5.support_1.query_15.lr_0.001.batch_size_16/
Namespace(log_dir='logs/protonet_successfull_way_5.support_1_and_5_query/omniglot.way_5.support_1.query_15.lr_0.001.batch_size_16/', num_way=5, num_support=5, num_query=15, learning_rate=0.001, batch_size=16, num_train_iterations=5000, test=True, checkpoint_step=4900, num_workers=2, compile=None, backend='inductor', cache=False, device='cpu')
Using device:  cpu
log_dir: logs/protonet_successfull_way_5.support_1_and_5_query/omniglot.way_5.support_1.query_15.lr_0.001.batch_size_16/
Loaded checkpoint iteration 4900.
Testing on tasks with composition num_way=5, num_support=5, num_query=15
Accuracy over 600 test tasks: mean 0.994, 95% confidence interval 0.001



#
# 5-Way, 1-Support
#

# Epoch 0

(XCS330) robr@Roberts-Laptop submission % python protonet.py --log_dir logs/protonet_successfull_way_5.support_1_and_5_query/omniglot.way_5.support_1.query_15.lr_0.001.batch_size_16/ --test --num_way 5 --num_support 1 --checkpoint_step 0 
Namespace(log_dir='logs/protonet_successfull_way_5.support_1_and_5_query/omniglot.way_5.support_1.query_15.lr_0.001.batch_size_16/', num_way=5, num_support=1, num_query=15, learning_rate=0.001, batch_size=16, num_train_iterations=5000, test=True, checkpoint_step=0, num_workers=2, compile=None, backend='inductor', cache=False, device='cpu')
Using device:  cpu
log_dir: logs/protonet_successfull_way_5.support_1_and_5_query/omniglot.way_5.support_1.query_15.lr_0.001.batch_size_16/
Loaded checkpoint iteration 0.
Testing on tasks with composition num_way=5, num_support=1, num_query=15
Accuracy over 600 test tasks: mean 0.585, 95% confidence interval 0.009

# Epoch 1000

(XCS330) robr@Roberts-Laptop submission % python protonet.py --log_dir logs/protonet_successfull_way_5.support_1_and_5_query/omniglot.way_5.support_1.query_15.lr_0.001.batch_size_16/ --test --num_way 5 --num_support 1 --checkpoint_step 1000
Namespace(log_dir='logs/protonet_successfull_way_5.support_1_and_5_query/omniglot.way_5.support_1.query_15.lr_0.001.batch_size_16/', num_way=5, num_support=1, num_query=15, learning_rate=0.001, batch_size=16, num_train_iterations=5000, test=True, checkpoint_step=1000, num_workers=2, compile=None, backend='inductor', cache=False, device='cpu')
Using device:  cpu
log_dir: logs/protonet_successfull_way_5.support_1_and_5_query/omniglot.way_5.support_1.query_15.lr_0.001.batch_size_16/
Loaded checkpoint iteration 1000.
Testing on tasks with composition num_way=5, num_support=1, num_query=15
Accuracy over 600 test tasks: mean 0.952, 95% confidence interval 0.004
(XCS330) robr@Roberts-Laptop submission % 


# Epoch 4900

(XCS330) robr@Roberts-Laptop submission % 
(XCS330) robr@Roberts-Laptop submission % python protonet.py --log_dir logs/protonet_successfull_way_5.support_1_and_5_query/omniglot.way_5.support_1.query_15.lr_0.001.batch_size_16/ --test --num_way 5 --num_support 1 --checkpoint_step 4900
Namespace(log_dir='logs/protonet_successfull_way_5.support_1_and_5_query/omniglot.way_5.support_1.query_15.lr_0.001.batch_size_16/', num_way=5, num_support=1, num_query=15, learning_rate=0.001, batch_size=16, num_train_iterations=5000, test=True, checkpoint_step=4900, num_workers=2, compile=None, backend='inductor', cache=False, device='cpu')
Using device:  cpu
log_dir: logs/protonet_successfull_way_5.support_1_and_5_query/omniglot.way_5.support_1.query_15.lr_0.001.batch_size_16/
Loaded checkpoint iteration 4900.
Testing on tasks with composition num_way=5, num_support=1, num_query=15
Accuracy over 600 test tasks: mean 0.978, 95% confidence interval 0.003


