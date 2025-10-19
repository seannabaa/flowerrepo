from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import base64
import io
from urllib import quote

app = Flask(__name__)

#load and preprocess data
df = pd.read_csv('Iris.csv')
df = df.drop(columns=['id'])

#Prepare data for models
X = df.drop(columns=['Species'])
Y = df['Species'].copy()

le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

#Train test split
x_train, x_test, y_train, y_test = train_test_split(X, Y_encoded, test_size=0.30, random_state=42)

#Train models
models = {
    'Logistic Regression' : LogisticRegression(max_iter=200),
    'K-nearest neighbors' : KNeighborsClassifier,
    'Decision Tree' : DecisionTreeClassifier
}

for model in models.values():
    model.fit(x_train, y_train)

def get_chart_image(fig):
    """Convert mataplotlib figure to base64 string"""
    img = io.BytesIO()
    fig.savefig(img, format="png", bbox_inches='tight')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return chart_url

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/data')
def data():
    stats = df.describe().to_html(classes='table table-striped')
    info = df.info()
    species_count = df['Species'].value_counts().to_dict()

    return render_template('data.html', 
                         stats=stats, 
                         species_count=species_count,
                         null_values=df.isnull().sum().to_dict())

@app.route('/visualization')
def visualization():
    charts = {}

    #Histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    df['SepalLengthCm'].hist(ax=axes[0, 0], bins=20, color='skyblue')
    axes[0, 0].set_title('Sepal Length Distribution')
    df['SepalWidthCm'].hist(ax=axes[0, 1], bins=20, color='lightcoral')
    axes[0, 1].set_title('Sepal Width Distribution')
    df['PetalLengthCm'].hist(ax=axes[1, 0], bins=20, color='lightgreen')
    axes[1, 0].set_title('Petal Length Distribution')
    df['PetalWidthCm'].hist(ax=axes[1, 1], bins=20, color='gold')
    axes[1, 1].set_title('Petal Width Distribution')
    plt.tight_layout()
    charts['histograms'] = get_chart_image(fig)

    #Scatter plots
    colors = {'Iris-virginica': 'red', 'Iris-versicolor': 'orange', 'Iris-setosa': 'blue'}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for species, color in colors.items():
        x = df[df['Species'] == species]
        axes[0, 0].scatter(x['SepalLengthCm'], x['SepalWidthCm'], c=color, label=species, alpha=0.6)
        axes[0, 1].scatter(x['PetalLengthCm'], x['PetalWidthCm'], c=color, label=species, alpha=0.6)
        axes[1, 0].scatter(x['SepalWidthCm'], x['PetalWidthCm'], c=color, label=species, alpha=0.6)
        axes[1, 1].scatter(x['SepalLengthCm'], x['PetalLengthCm'], c=color, label=species, alpha=0.6)
    
    axes[0, 0].set(xlabel='Sepal Length', ylabel='Sepal Width', title='Sepal Dimensions')
    axes[0, 1].set(xlabel='Petal Length', ylabel='Petal Width', title='Petal Dimensions')
    axes[1, 0].set(xlabel='Sepal Width', ylabel='Petal Width', title='Width Comparison')
    axes[1, 1].set(xlabel='Sepal Length', ylabel='Petal Length', title='Length Comparison')
    
    for ax in axes.flat:
        ax.legend()
    plt.tight_layout()
    charts['scatter'] = get_chart_image(fig)
    
    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df.drop(columns=['Species']).corr()
    sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    charts['correlation'] = get_chart_image(fig)
    
    return render_template('visualization.html', charts=charts)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    input_data = np.array([[
        float(data['sepal_length']),
        float(data['sepal_width']),
        float(data['petal_length']),
        float(data['petal_width'])
    ]])

    model_name = data['model']
    model = models[model_name]

    prediction = model.predict(input_data)[0]
    species = le.inverse_transform([prediction])[0]

    probability = model.predict_proba(input_data)[0]
    probabilities = {le.classes[i]: round(float(probability[i]) * 100, 2) for i in range(len(le.classes_)) }

    return jsonify({
        'prediction' : species,
        'probabilities' : probabilities
    })

if __name__ == "__main__":
    app.run(debug=True)

        
