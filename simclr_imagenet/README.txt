A. saved
    - simclr1x_lg_1000_classes_model.sav (OneVsRestClassifier with logistic regression)
    - test_pred_prob_1000_classes.npy (saved prediction probability for testing data)
    - test_X_1000_classes.npy (X of testing dataset)
    - test_y_1000_classes.npy (y of testing dataset)
    - train_X_1000_classes.npy (X of training dataset)
    - train_y_1000_classes.npy (y of training dataset)

===========================================================================================================

B. scripts
    - simclr_uniform_exp.py (simclr embeddings + onevsrest + logistic regression + uniform sampler)
    - simclr_exponential_exp.py (simclr embeddings + onevsrest + logistic regression + exponential sampler)
    - train_1000.py (this is similar to simclr_uniform_exp, but I made this to saved the trained logistic regression -- for saving time cause training lg is time-consuming)
    - exponential_load_1000lg_extra_exp.py (this is a script to conduct experiment with more sampled classes and see performance change)
    - uniform_load_1000lg_extra_exp.py (this is a script to conduct experiment with more sampled classes and see performance change)
    - resnet_wider.py (to call simclr, can be ignored)
    
===========================================================================================================

C. notebooks:
    - load_saved_model.ipynb (notebook version of uniform_load_1000lg_extra_exp and exponential_load_1000lg_extra_exp)
    - simclr_exponential.ipynb (notebook version of simclr_exponential_exp)
    - simclr_uniform.ipynb (notebook version of simclr_uniform_exp)
    - resnet_wider.py (to call simclr, can be ignored)
===========================================================================================================

D. others:
    - dir_label_name.json (a table to link class number, class id, and class name in minitree)
    - imagenet_mintree.txt (class structure)
    - command_template.sh (command examples for u to look)
    - resnet50-1x.pth (saved simclr, in paper, we use this version)
    - resnet50-2x.pth (saved simclr)
    - resnet50-4x.pth (saved simclr)
    
===========================================================================================================
To run simclr_uniform_exp, command should be:
## python3 program_name num_of_sampled_classes structure per_class seed resnet?x.pth
for example like,
# python3 simclr_uniform_exp.py 10 "mintree" 50 123 1
# python3 simclr_uniform_exp.py 50 "mintree" 50 123 1

To run simclr_exponential_exp, command should be:
## python3 program_name num_of_sampled_classes num_of_around_center structure per_class seed resnet?x.pth
for example like,
python3 simclr_exponential_exp.py 10 10 "mintree" 50 123 1
python3 simclr_exponential_exp.py 50 10 "mintree" 50 123 1
