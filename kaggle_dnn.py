from utilities import *

train_data,label = load_data(train_data = 1)
test = load_data(train_data=0)

train_data,test = preprocess_data(train_data,test)
train_data = z_score(train_data)
train_data = np.tanh(train_data)
# plt.hist(train_data)
# plt.show()
label = label.ravel()
test = z_score(test)
test = np.tanh(test)

# for i in range(2,50,2):
#     for j in range(50,100,10):
#         clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=i), n_estimators = j)
#         scores = cross_val_score(clf,X,y,cv =3)
#         print 'maxdepth is ', i, 'n_esti is: ',j, 'score is:',scores
feature_filter = SelectKBest(f_classif,k =100)
pca = KernelPCA(kernel='rbf',fit_inverse_transform=True,gamma=10)
# anova_filter = PCA(n_components=10)


# clf = RandomForestClassifier()
# clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20), n_estimators=15)
# clf = svm.SVC(kernel='sigmoid')
clf = AdaBoostClassifier(ExtraTreesClassifier(max_depth=20),n_estimators=70)
# clf = MLPClassifier(solver = 'adam',alpha=1e-4,hidden_layer_sizes=(50,50),random_state=1)

anova_svm = Pipeline([('feature_filter',feature_filter),('ada',clf)])
# anova_svm.set_params(anova__k = 200)
anova_svm.fit(train_data,label)
scores = cross_val_score(anova_svm, train_data, label, cv=5)
print scores

result =  anova_svm.predict(test)
write_to_excel(result)