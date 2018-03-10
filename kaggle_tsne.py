

train_data,label = load_data()
train_data[152][68] = train_data[152][68]/1000
feature_selection = range(train_data.shape[1])

index2remove = np.array([6,7,14,15,24,25,38,39,40,41,42,43,44,71])-3
index2mul1000 =np.array(range(69,139) + range(277,413))
index2div1000 = np.array(range(413,429))


for i in index2mul1000:
    for j in range(train_data.shape[0]):
        if train_data[j,i] < 10:
            train_data[j,i] = train_data[j,i] * 1000.0

for i in index2div1000:
    for j in range(train_data.shape[0]):
        if train_data[j, i] > 10000:
            train_data[j, i] = train_data[j, i] / 1000.0

for i in range(len(index2remove)):
    feature_selection.remove(index2remove[i])

# for col_index in range(train_data.shape[1]):
#     # train_data[:,col_index] = rfe_sigma(train_data[:,col_index],sigma =2)
#     # train_data[:,col_index] = median(train_data[:,col_index])  # use median
#     # temp = z_score(train_data[:,col_index])
#


train_data = train_data[:,feature_selection]
# train_data = np.array([train_data])
train_data = z_score(train_data)
train_data = np.tanh(train_data)
# train_data = preprocessing.scale(train_data)  # standard the train_data
# plt.plot(train_data)
# plt.show()
# a = raw_input('input')
# col_to_remove = ~np.isnan(train_data)[0] # remove nan train_data
# for i in range(len(col_to_remove)):
#     if col_to_remove[i] == False:
#



X_new = SelectKBest(f_regression,k= 10).fit_transform(train_data,label)
# tsne
names = [0,1,2,3]

# choose_perplexity_tsne()
fix_perplexity_tsne()

