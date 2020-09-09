import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

def main():
	st.title("Binary Classification App Made By Kaushik Bhide TCOD62")
	st.sidebar.title("Binary Classifcation App")
	st.markdown("Are Mushrooms Poisonous or Edible?🍄 Right Click to See Options")
	st.sidebar.markdown("Are Mushrooms Poisonous or Edible?🍄")
	@st.cache(persist=True)
	def load_data():
		data = pd.read_csv('mushrooms.csv')
		label = LabelEncoder()
		for col in data.columns:
			data[col] = label.fit_transform(data[col])
		return data
	def split(df):
		y = df.type
		x = df.drop(columns=['type'])
		x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
		return x_train, x_test , y_train, y_test

	def plot_metrics(metrics_list):
		if 'Confusion Matrix' in metrics_list:
			st.subheader("Confusion Matrix")
			plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
			st.pyplot()

		if 'ROC Curve' in metrics_list:
			st.subheader("ROC Curve")
			plot_roc_curve(model, x_test, y_test)
			st.pyplot()

		if 'Precision-Recall-Curve' in metrics_list:
			st.subheader("Precision-Recall-Curve")
			plot_precision_recall_curve(model, x_test, y_test)
			st.pyplot()
		
	df = load_data()
	x_train,x_test,y_train,y_test = split(df)
	class_names = ['Edible','Poisonous']
	st.sidebar.subheader("Choose Classifier")
	classifier = st.sidebar.selectbox("Available Classifiers",("Support Vector Machine(SVM)","LogisticRegression(LR)","RandomForest(RF)"))

	if classifier == 'Support Vector Machine(SVM)':
		st.sidebar.subheader('Model Hyperparameters')
		C = st.sidebar.number_input("C (Regularization parameter)",0.01,10.0, step=0.01,key='C')
		kernel = st.sidebar.radio("Kernel",("rbf","linear"),key='kernel')
		gamma = st.sidebar.radio("Gamma(Kernel Coefficient)",("scale","auto"),key='gamma')

		metrics = st.sidebar.multiselect("Choose Metrics To Plot",('Confusion Matrix','ROC Curve','Precision-Recall-Curve'))

		if st.sidebar.button("Classify", key='classify'):
			st.subheader("Support Vector Machine Results")
			model = SVC(C=C,kernel=kernel,gamma=gamma)
			model.fit(x_train,y_train)
			accuracy = model.score(x_test,y_test)
			y_pred = model.predict(x_test)
			st.write("Accuracy:", accuracy.round(2))
			st.write("Precision:", precision_score(y_test,y_pred,labels=class_names).round(2))
			st.write("Recall:", recall_score(y_test,y_pred,labels=class_names).round(2))
			plot_metrics(metrics)

	if classifier == 'RandomForest(RF)':
		st.sidebar.subheader('Model Hyperparameters')
		n_estimators = st.sidebar.number_input("The Number of Trees in the forest", 100, 5000, step =10 , key='n_estimators')
		max_depth = st.sidebar.number_input("Maximum Depth", 1,20, step=2, key='max_depth')
		bootstrap = st.sidebar.radio("bootstrap samples when building trees", ("True","False"), key='bootstrap')

		metrics = st.sidebar.multiselect("Choose Metrics To Plot",('Confusion Matrix','ROC Curve','Precision-Recall-Curve'))
		

		if st.sidebar.button("Classify", key='classify'):
			st.subheader("LogisticRegression Results")
			model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap, n_jobs=-1)
			model.fit(x_train,y_train)
			accuracy = model.score(x_test,y_test)
			y_pred = model.predict(x_test)
			st.write("Accuracy:", accuracy.round(2))
			st.write("Precision:", precision_score(y_test,y_pred,labels=class_names).round(2))
			st.write("Recall:", recall_score(y_test,y_pred,labels=class_names).round(2))
			plot_metrics(metrics)



	if classifier == 'LogisticRegression(LR)':
		st.sidebar.subheader('Model Hyperparameters')
		C = st.sidebar.number_input("C (Regularization parameter)",0.01,10.0, step=0.01,key='C_LR')
		max_iter = st.sidebar.slider("Maximum Number of Iterations",100,500, key='max_iter')

		metrics = st.sidebar.multiselect("Choose Metrics To Plot",('Confusion Matrix','ROC Curve','Precision-Recall-Curve'))

		if st.sidebar.button("Classify", key='classify'):
			st.subheader("LogisticRegression Results")
			model = SVC(C=C, max_iter=max_iter)
			model.fit(x_train,y_train)
			accuracy = model.score(x_test,y_test)
			y_pred = model.predict(x_test)
			st.write("Accuracy:", accuracy.round(2))
			st.write("Precision:", precision_score(y_test,y_pred,labels=class_names).round(2))
			st.write("Recall:", recall_score(y_test,y_pred,labels=class_names).round(2))
			plot_metrics(metrics)







	if st.sidebar.checkbox("Show Raw Data", True):
		st.subheader("Mushroom Data Set(Classifcation)")
		st.write(df)


if __name__ == '__main__':
    main()


