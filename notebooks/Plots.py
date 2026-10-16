import matplotlib as mlp
import matplotlib.pyplot as plt
import pickle
import scienceplots
import numpy as np

fig = plt.figure(dpi = 300)
plt.style.use("science")
plt.figure(figsize=(10,10))
plt.rc('font', size=20)


# Accuracy
pkl_filename = "normal_normal_train_val_accuracy_loss.csv"
with open(pkl_filename, 'rb') as file:
    acc_loss = pickle.load(file)


pkl_filename = "normal_train_val_accuracy_loss.csv"
with open(pkl_filename, 'rb') as file:
    bad_acc_loss = pickle.load(file)


train_accuracy = [a[0] for a in acc_loss]
val_accuracy = [a[1] for a in acc_loss]


plt.plot(train_accuracy, color = "C0",label='Train accuracy')
plt.plot(val_accuracy, color = "C1", label = "Val accuracy")


train_accuracy = [a[0] for a in bad_acc_loss]
val_accuracy = [a[1] for a in bad_acc_loss]

plt.plot(train_accuracy,linestyle = "--", color = "C0",label='Train accuracy (Noisy labels)')
plt.plot(val_accuracy,linestyle = "--",  color = "C1", label = "Val accuracy (Noisy labels)")


plt.ylabel('Model accuracy')
plt.xlabel('Epoch')


plt.savefig('Train_test_accuracy.pdf', dpi=300) 




train_loss = [a[2].item()/acc_loss[1:][0][2].item() for a in acc_loss[1:]]
val_loss = [a[3].item()/acc_loss[1:][0][3].item()  for a in acc_loss[1:]]


plt.plot(train_loss, color = "C0",label='Train loss')
plt.plot(val_loss, color = "C1", label = "Val loss")


train_loss = [a[2].item()/bad_acc_loss[1:][0][2].item() for a in bad_acc_loss[1:]]
val_loss = [a[3].item()/bad_acc_loss[1:][0][3].item()  for a in bad_acc_loss[1:]]

plt.plot(train_loss,linestyle = "--", color = "C0",label='Train loss (Noisy labels)')
plt.plot(val_loss,linestyle = "--",  color = "C1", label = "Val loss (Noisy labels)")


plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend()



plt.savefig('Train_test_loss,pdf', dpi=300) 




pkl_filename = "normal_normal_train_test_TPCF_score.csv"
with open(pkl_filename, 'rb') as file:
    TPCF_scores = pickle.load(file)


pkl_filename = "normal_train_test_TPCF_score.csv"
with open(pkl_filename, 'rb') as file:
    bad_TPCF_scores = pickle.load(file)



train_TPCF = [a[0][0] for a in TPCF_scores]
train_TPCF_error = [a[0][1] for a in TPCF_scores]

val_TPCF = [a[1][0] for a in TPCF_scores]
val_TPCF_error = [a[1][1] for a in TPCF_scores]



#plt.plot(train_TPCF)
plt.plot(train_TPCF, color = "C0",label='Train TPCF score')

plt.fill_between(x,[a-b for a,b in zip(train_TPCF,train_TPCF_error)], [a+b for a,b in zip(train_TPCF,train_TPCF_error)], color="C0", alpha=0.1)
plt.plot(val_TPCF, color = "C1", label = "Val TPCF score")

plt.fill_between(x,[a-b for a,b in zip(val_TPCF,val_TPCF_error)], [a+b for a,b in zip(val_TPCF,val_TPCF_error)], color="C1", alpha=0.1)


bad_train_TPCF = [a[0][0] for a in bad_TPCF_scores]
bad_train_TPCF_error = [a[0][1] for a in bad_TPCF_scores]

bad_val_TPCF = [a[1][0] for a in bad_TPCF_scores]
bad_val_TPCF_error = [a[1][1] for a in bad_TPCF_scores]



#plt.plot(train_TPCF)
plt.plot(bad_train_TPCF, color = "C0",linestyle = "--",label='Train TPCF score')
plt.fill_between(x,[a-b for a,b in zip(bad_train_TPCF,bad_train_TPCF_error)], [a+b for a,b in zip(bad_train_TPCF,bad_train_TPCF_error)], color="C0", alpha=0.1)


plt.plot(bad_val_TPCF, color = "C1",linestyle = "--", label = "Val TPCF score")
plt.fill_between(x,[a-b for a,b in zip(bad_val_TPCF,bad_val_TPCF_error)], [a+b for a,b in zip(bad_val_TPCF,bad_val_TPCF_error)], color="C1", alpha=0.1)

plt.ylabel('TPCF score')
plt.xlabel('Epoch')
plt.legend()

plt.savefig('Train_test_id_score,pdf', dpi=300) 



pkl_filename = "normal_normal_train_val_id_score.csv"
with open(pkl_filename, 'rb') as file:
    id_scores = pickle.load(file)


pkl_filename = "normal_train_val_id_score.csv"
with open(pkl_filename, 'rb') as file:
    bad_id_scores = pickle.load(file)



train_id = [a[0] for a in id_scores]
val_id = [a[2] for a in id_scores]


plt.plot(train_id, color = "C0",label='Train ID score')
plt.plot(val_id, color = "C1", label = "Val ID score")


bad_train_id = [a[0] for a in bad_id_scores]
bad_val_id = [a[2] for a in bad_id_scores]


plt.plot(bad_train_id,linestyle = "--", color = "C0",label='Train ID score (Noisy labels)')

plt.plot(bad_val_id,linestyle = "--",  color = "C1", label = "Val ID score (Noisy labels)")


plt.ylabel('Intrinsic Dimension score')
plt.xlabel('Epoch')

plt.legend()


plt.savefig('Train_test_id_score,pdf', dpi=300) 
plt.show()