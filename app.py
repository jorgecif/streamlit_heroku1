# Core pkgs
import streamlit as st



# EDA pkgs
import pandas as pd 
import numpy as np

# Data visulization pkgs
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
st.set_option('deprecation.showfileUploaderEncoding', False)  # Apagar warning

# ML Pkgs
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def main():
	"""Aplicación de aprendizaje automático"""

	st.title("Aplicación de aprendizaje automático")
	st.text("Exploración de datos, visualización y construcción del modelo")

	activites = ["Análisis exploratorio","Visualización","Modelos","Créditos"]
	activites2 = ["Análisis exploratorio","Visualización","Modelos","Créditos"]


	choice = st.sidebar.selectbox("Seleccione",activites)

	if choice == 'Análisis exploratorio':
		st.subheader("Análisis exploratorio de los datos")

		data = st.file_uploader("Subir Dataset",type=["csv","txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())

			if st.checkbox("Ver dimensiones"):
				st.write(df.shape)

			if st.checkbox("Mostrar columnas"):
				all_columns = df.columns.to_list()
				st.write(all_columns)

			if st.checkbox("Mostrar resumen"):
				st.write(df.describe())

			if st.checkbox("Seleccionar columnas a mostrar"):
				all_columns = df.columns.to_list()
				selected_columns = st.multiselect("Select Columns",all_columns)
				new_df =  df[selected_columns]
				st.dataframe(new_df)

			if st.checkbox("Conteo"):
				st.write(df.iloc[:,-1].value_counts())

			if st.checkbox("Correlaciones"):
				st.write(sns.heatmap(df.corr(),annot=True))
				st.pyplot()

			if st.checkbox("Gráfica de pie"):
				all_columns = df.columns.to_list()
				columns_to_plot = st.selectbox("Select 1 Column ",all_columns)
				pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
				st.write(pie_plot)
				st.pyplot()

	elif choice == 'Visualización':
		st.subheader("Visualización de datos")
		data = st.file_uploader("Subir Dataset",type=["csv","txt"])

		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())

		all_columns_names = df.columns.tolist()
		type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
		selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

		if st.button("Generar gráfico"):
			st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

			# Plot By Streamlit
			if type_of_plot == 'area':
				cust_data = df[selected_columns_names]
				st.area_chart(cust_data)

			elif type_of_plot == 'bar':
				cust_data = df[selected_columns_names]
				st.bar_chart(cust_data)

			elif type_of_plot == 'line':
				cust_data = df[selected_columns_names]
				st.line_chart(cust_data)

			# Custom Plot 
			elif type_of_plot:
				cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
				st.write(cust_plot)
				st.pyplot()



	elif choice == 'Modelos':
		st.subheader("Evaluación de diferentes modelos de Machine Learning")

		data = st.file_uploader("Subir Dataset",type=["csv","txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())

		# Model Building
		X = df.iloc[:,0:-1]
		Y = df.iloc[:,-1]
		seed = 7

		# Model 
		models = []
		models.append(("LR",LogisticRegression()))
		models.append(("LDA",LinearDiscriminantAnalysis()))
		models.append(("KNN",KNeighborsClassifier()))
		models.append(('CART', DecisionTreeClassifier()))
		models.append(('NB', GaussianNB()))
		models.append(('SVM', SVC()))
		# evaluate each model in turn

		# List
		model_names = []
		model_mean = []
		model_std = []
		all_models = []
		scoring = 'accuracy'

		for name,model in models:
			kfold = model_selection.KFold(n_splits=10, random_state=seed)
			cv_results = model_selection.cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
			model_names.append(name)
			model_mean.append(cv_results.mean())
			model_std.append(cv_results.std())

			accuracy_results = {"model_name":name,"model_accuracy":cv_results.mean(),"standard_deviation":cv_results.std()}
			all_models.append(accuracy_results)

		if st.checkbox("Métricas como tabla"):
			st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns=["Model Name","Model Accuracy","Standard Deviation"]))

		if st.checkbox("Métricas como JSON"):
			st.json(all_models)




	elif choice == 'Créditos':
		st.subheader("Créditos")
		st.text("Jorge O. Cifuentes")
		st.write('*jorgecif@gmail.com* :sunglasses:')




if __name__ == "__main__":
    main()