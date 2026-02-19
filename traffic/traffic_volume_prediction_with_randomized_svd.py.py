# Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt
import time
import matplotlib.dates as mdates

# Plantamos semilla
np . random . seed (42)

# Cargar los datos
data = pd.read_csv("TrafficVolumeData.csv")
# Se elimina la variable "dew_point" debido a que por error en la toma de datos se duplicó los valores de la varible "visibility_in_miles"
data=data.drop(columns=['dew_point'])
print(data)
numericas = data.select_dtypes(include=["number"])
summary = numericas.describe(include='all')
print(summary)

### 1. Análisis Exploratorio de los Datos (EDA)
# Convertir temperatura de Kelvin a Celsius
data2=data
data2['temp_celsius'] = data['temperature'] - 273.15


#Grafico para ver las proporciones entre dias festivos y laborables
holiday_counts = [data['is_holiday'].value_counts().sum(), data['is_holiday'].isna().sum()]
df = pd.DataFrame(holiday_counts, columns=['Vacaciones'])

# Gráfico de pastel con porcentajes y valores numéricos
plt.figure(figsize=(6, 6))

total = sum(df['Vacaciones'])  # Suma total de los valores
df['Vacaciones'].plot(
    kind='pie',
    labels=['Días festivos', 'Días no festivos'], 
    colors=['skyblue', 'orange'], 
    autopct=lambda pct: f"{pct:.1f}%\n{int(round(pct/100. * total))}",  # Formateo directo
    startangle=90
)
plt.title('Proporción de Días Feriados')
plt.ylabel('')  # Ocultar la etiqueta del eje Y
plt.show()

# Gráfico Distribución de 'is_holiday' 
plt.figure(figsize=(10, 5))
ax = sns.countplot(x='is_holiday', data=data, palette='viridis', order=data['is_holiday'].value_counts().index)
plt.title('Frecuencia de los días festivos', fontsize=14)
plt.xticks(rotation=15)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
plt.savefig('grafico_is_holiday_frecuencia.png')
plt.show()


# Gráfico de la distribución de 'air_pollution_index'
plt.figure(figsize=(10, 5))
sns.histplot(data2['air_pollution_index'], kde=True, bins=60, color='blue', edgecolor='black')
plt.axvline(data2['air_pollution_index'].mean(), color='red', linestyle='--', 
            label=f"Media: {data2['air_pollution_index'].mean():.2f}")
plt.axvline(data2['air_pollution_index'].quantile(0.25), color='orange', linestyle='--', 
            label=f"Q1: {data2['air_pollution_index'].quantile(0.25):.2f}")
plt.axvline(data2['air_pollution_index'].quantile(0.5), color='green', linestyle='--', 
            label=f"Mediana: {data2['air_pollution_index'].quantile(0.5):.2f}")
plt.axvline(data2['air_pollution_index'].quantile(0.75), color='purple', linestyle='--', 
            label=f"Q3: {data2['air_pollution_index'].quantile(0.75):.2f}")
sns.kdeplot(data2['air_pollution_index'], color='blue', linewidth=2, label='Función de Densidad')
plt.title('Distribución del Índice de Contaminación del Aire', fontsize=14)
plt.xlabel('Índice de Contaminación del Aire', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.legend()
plt.savefig('grafico_air_pollution_index_distribucion.png')
plt.show()



# Gráfico de la distribución de 'humidity'
plt.figure(figsize=(10, 5))
sns.histplot(data2['humidity'], kde=True, bins=60, color='blue', edgecolor='black')
plt.axvline(data2['humidity'].mean(), color='red', linestyle='--', 
            label=f"Media: {data2['humidity'].mean():.2f}")
plt.axvline(data2['humidity'].quantile(0.25), color='orange', linestyle='--', 
            label=f"Q1: {data2['humidity'].quantile(0.25):.2f}")
plt.axvline(data2['humidity'].quantile(0.5), color='green', linestyle='--', 
            label=f"Mediana: {data2['humidity'].quantile(0.5):.2f}")
plt.axvline(data2['humidity'].quantile(0.75), color='purple', linestyle='--', 
            label=f"Q3: {data2['humidity'].quantile(0.75):.2f}")
sns.kdeplot(data2['humidity'], color='blue', linewidth=2, label='Función de Densidad')
plt.title('Distribución del Índice de humidity', fontsize=14)
plt.xlabel('Índice de humidity', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.legend()
plt.savefig('grafico_humidity_distribucion.png')
plt.show()

# Gráfico de la distribución de 'wind_speed'
plt.figure(figsize=(10, 5))
ax = sns.countplot(x='wind_speed', data=data, palette='viridis', order=data['wind_speed'].value_counts().index)
plt.title('Frecuencia de los niveles de viento', fontsize=14)

plt.xticks(rotation=15)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
plt.savefig('grafico_wind_speed_frecuencia.png')
plt.show()

#grafico de direccion del viento
plt.figure(figsize=(10, 5))
sns.histplot(data2['wind_direction'], kde=True, bins=60, color='blue', edgecolor='black')
plt.axvline(data2['wind_direction'].mean(), color='red', linestyle='--', 
            label=f"Media: {data2['wind_direction'].mean():.2f}")
plt.axvline(data2['wind_direction'].quantile(0.25), color='orange', linestyle='--', 
            label=f"Q1: {data2['wind_direction'].quantile(0.25):.2f}")
plt.axvline(data2['wind_direction'].quantile(0.5), color='green', linestyle='--', 
            label=f"Mediana: {data2['wind_direction'].quantile(0.5):.2f}")
plt.axvline(data2['wind_direction'].quantile(0.75), color='purple', linestyle='--', 
            label=f"Q3: {data2['wind_direction'].quantile(0.75):.2f}")
sns.kdeplot(data2['wind_direction'], color='blue', linewidth=2, label='Función de Densidad')
plt.title('Distribución de la dirección del viento', fontsize=14)
plt.xlabel('Ángulos', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.legend()
plt.savefig('grafico_wind_direction_distribucion.png')
plt.show()

# Gráfico de la distribución de 'visibility_in_miles'
plt.figure(figsize=(10, 5))
ax = sns.countplot(x='visibility_in_miles', data=data, palette='viridis', order=data['visibility_in_miles'].value_counts().index)
plt.title('Frecuencia de millas visibles', fontsize=14)
plt.xticks(rotation=15)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
plt.savefig('grafico_visibility_in_miles_frecuencia.png')
plt.show()

# Gráfico Distribución de 'clouds_all' 
plt.figure(figsize=(10, 5))
sns.histplot(data['clouds_all'], kde=True, bins=30, color='grey', edgecolor='black')
plt.axvline(data['clouds_all'].mean(), color='red', linestyle='--', label=f"Media: {data['clouds_all'].mean():.2f}")
plt.axvline(data['clouds_all'].quantile(0.25), color='green', linestyle='--', label=f"Q1: {data['clouds_all'].quantile(0.25):.2f}")
plt.axvline(data['clouds_all'].quantile(0.75), color='purple', linestyle='--', label=f"Q3: {data['clouds_all'].quantile(0.75):.2f}")
plt.title('Distribución de la Cobertura de Nubes (%)', fontsize=14)
plt.xlabel('Cobertura de Nubes (%)', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.xlim(data['clouds_all'].min(), data['clouds_all'].max())

plt.legend()
plt.savefig('grafico_clouds_all_distribucion.png')
plt.show()

# Gráfico Distribución de 'weather_type' 
plt.figure(figsize=(10, 5))
ax = sns.countplot(x='weather_type', data=data, palette='coolwarm', order=data['weather_type'].value_counts().index)
plt.title('Frecuencia de Weather Main', fontsize=14)
plt.xlabel('Weather Main', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)

plt.xticks(rotation=45)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
plt.savefig('grafico_weather_type_frecuencia.png')
plt.show()

# Gráfico Distribución de 'weather_description' 
plt.figure(figsize=(12, 10))  # Ampliar para mejorar legibilidad
data['weather_description'] = data['weather_description'].str.capitalize()
ax = sns.countplot(y='weather_description', data=data, palette='magma', 
              order=data['weather_description'].value_counts().index)
plt.title('Frecuencia de Weather Description', fontsize=14)
plt.xlabel('Frecuencia', fontsize=12)
plt.ylabel('Weather Description', fontsize=12)
plt.yticks(fontsize=7, rotation=15)  # Reducir tamaño de la letra en el eje Y y ajustar orientación
for p in ax.patches:
    ax.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2.), ha='left', va='center', fontsize=8, color='black', xytext=(5, 0), textcoords='offset points')
plt.savefig('grafico_weather_description_frecuencia.png')
plt.show()

# Gráfico de 'weather_description' palabras presentación
from wordcloud import WordCloud

text = ' '.join(data['weather_description'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de Palabras - Descripciones Climáticas')
plt.show()

# Gráfico Distribución de 'traffic_volume' 
plt.figure(figsize=(10, 5))
sns.histplot(data['traffic_volume'], kde=True, bins=30, color='red', edgecolor='black')
plt.axvline(data['traffic_volume'].mean(), color='blue', linestyle='--', label=f"Media: {data['traffic_volume'].mean():.2f}")
plt.axvline(data['traffic_volume'].median(), color='green', linestyle='--', label=f"Mediana: {data['traffic_volume'].median():.2f}")
plt.axvline(data['traffic_volume'].quantile(0.25), color='orange', linestyle='--', label=f"Q1: {data['traffic_volume'].quantile(0.25):.2f}")
plt.axvline(data['traffic_volume'].quantile(0.75), color='purple', linestyle='--', label=f"Q3: {data['traffic_volume'].quantile(0.75):.2f}")
plt.title('Distribución de Traffic Volume', fontsize=14)
plt.xlabel('Traffic Volume', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.xlim(data['traffic_volume'].min(), data['traffic_volume'].max())

plt.legend()
plt.savefig('grafico_traffic_volume_distribucion.png')
plt.show()

#multivariantes
# Grafico relación tráfico y tiempo en dias:
dia_filtrado = data.head(48)

plt.figure(figsize=(10, 6))
plt.plot(dia_filtrado["date_time"], dia_filtrado["traffic_volume"], marker='o', color='blue')
plt.title("Tráfico por Hora (Primer Día de Datos)", fontsize=14)
plt.xlabel("Hora", fontsize=12)
plt.ylabel("Volumen de Tráfico", fontsize=12)
plt.xticks(rotation=75)  
plt.tight_layout()
plt.savefig('grafico_trafico_dia.png')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='weather_type', y='traffic_volume')
plt.title('Relación entre Traffic_volume y Weather_type')
plt.xticks(rotation=45)
plt.savefig('grafico_trafico_clima.png')
plt.show()



### 2. Desarrollo del Modelo de Predicción

# Convertir `date_time` a formato numérico
data['date_time'] = pd.to_datetime(data['date_time']).astype(int) / 10**9

# Convertir las columnas categóricas a valores numéricos (codificación ordinal)
categorical_cols = ['is_holiday', 'weather_type', 'weather_description']
data[categorical_cols] = data[categorical_cols].astype('category').apply(lambda x: x.cat.codes)

# Calcular la matriz de correlación únicamente con las variables originales
correlation_matrix = data.corr()

# Seleccionar solo las correlaciones con la columna `traffic_volume`
traffic_volume_correlation = correlation_matrix['traffic_volume']

# Filtrar solo las 13 variables originales 
original_columns = ['date_time', 'air_pollution_index', 'humidity', 'wind_speed', 
                    'wind_direction', 'visibility_in_miles', 'temperature',
                    'rain_p_h', 'snow_p_h', 'clouds_all', 'weather_type', 'weather_description', 'is_holiday', 'traffic_volume']
traffic_volume_correlation = traffic_volume_correlation.loc[original_columns]


# Imprimir las correlaciones de `traffic_volume`
print(traffic_volume_correlation)

# Visualizar las correlaciones en un heatmap
plt.figure(figsize=(8, 10))
sns.heatmap(traffic_volume_correlation.to_frame(), cmap="coolwarm", annot=True, fmt=".3f", cbar=True)
plt.title("Correlations of Variables with Traffic Volume")
plt.show()



# Separar variables predictoras y objetivo
X = data.drop(columns=['traffic_volume'])
y = data['traffic_volume']

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Transformar a características polinómicas de grado 3
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Aplicar SVD aleatorizado para reducir la dimensionalidad
n_components = 2
U, Sigma, VT = randomized_svd(X_train_poly, n_components=n_components, random_state=42)

# Construir la matriz reducida B
B = np.dot(U, np.diag(Sigma))

# Comparación de varianza explicada entre la matriz original y reducida
original_variance = np.var(X_train_poly, axis=0).sum()
sketched_variance = np.var(B, axis=0).sum()
explained_variance_ratio = (sketched_variance / original_variance) * 100
print(f"Varianza explicada por la matriz reducida: {explained_variance_ratio:.3f}\n")

# Evaluar el modelo normal
start_time = time.time()
model_normal = LinearRegression()
model_normal.fit(X_train_poly, y_train)
y_pred_normal = model_normal.predict(X_test_poly)
end_time = time.time()

mae_normal = mean_absolute_error(y_test, y_pred_normal)
mse_normal = mean_squared_error(y_test, y_pred_normal)
r2_normal = r2_score(y_test, y_pred_normal)
time_normal = end_time - start_time

# Evaluar el modelo con sketching
start_time = time.time()
model_sketch = LinearRegression()
model_sketch.fit(B, y_train)
X_test_sketch = np.dot(X_test_poly, VT.T)
y_pred_sketch = model_sketch.predict(X_test_sketch)
end_time = time.time()

mae_sketch = mean_absolute_error(y_test, y_pred_sketch)
mse_sketch = mean_squared_error(y_test, y_pred_sketch)
r2_sketch = r2_score(y_test, y_pred_sketch)
time_sketch = end_time - start_time

# Mostrar resultados 
print("\nResultados del modelo normal:")
print(f"MAE: {mae_normal:.3e}")
print(f"MSE: {mse_normal:.3e}")
print(f"R²: {r2_normal:.3e}")
print(f"Tiempo de entrenamiento y predicción: {time_normal:.3f} segundos\n")

# Error Absoluto Medio (MAE): 4.327×10^8, lo que indica un error de predicción extremadamente alto.Esto sugiere que las predicciones del modelo están muy alejadas de los valores reales.

# Error Cuadrático Medio (MSE): 4.390×10^20, lo que refuerza la idea de que los errores son masivos y que las predicciones del modelo normal son inadecuadas para los datos.

# Coeficiente de Determinación (R^2):−1.099×10^14, un valor negativo extremadamente grande. Un R^2 negativo indica que el modelo es peor que un modelo que simplemente predice el promedio de los datos.

# Tiempo de Ejecución: 1.716 segundos. Aunque no es excesivamente lento, el tiempo de ejecución es significativamente mayor que el del modelo Sketching.

#En resumen, el modelo normal parece no ajustarse correctamente a los datos y presenta predicciones altamente erroneas.

print("Resultados del modelo con sketching:")
print(f"MAE: {mae_sketch:.3e}")
print(f"MSE: {mse_sketch:.3e}")
print(f"R²: {r2_sketch:.3e}")
print(f"Tiempo de entrenamiento y predicción: {time_sketch:.3f} segundos\n")


# El modelo Sketching muestra un rendimiento mucho mejor en comparación con el modelo normal:

# Error Absoluto Medio (MAE): 1.759×10^3, lo que indica que las predicciones están mucho más cerca de los valores reales en comparación con el modelo normal.

# Error Cuadrático Medio (MSE): 3.994×10^6, un valor mucho más bajo que el del modelo normal, sugiriendo un ajuste considerablemente más preciso.

#Coeficiente de Determinación (R^2):−5.032×10^−6, aunque sigue siendo ligeramente negativo (indicando que no es perfecto),
# el valor está muy cerca de cero, lo que implica un rendimiento razonable en comparación con el modelo normal.

# Tiempo de Ejecución: 0.008 segundos. El modelo Sketching es extremadamente rápido, completando el proceso en una fracción de tiempo comparado con el modelo normal.

# Comparación en tabla
print("\nComparación de modelos:")
print("-" * 80)
print(f"{'Modelo':<20} | {'MAE':^12} | {'MSE':^12} | {'R²':^12} | {'Tiempo (s)':^12}")
print("-" * 80)
print(f"{'Normal':<20} | {mae_normal:^12.3e} | {mse_normal:^12.3e} | {r2_normal:^12.3e} | {time_normal:^12.4f}")
print(f"{'Sketching':<20} | {mae_sketch:^12.3e} | {mse_sketch:^12.3e} | {r2_sketch:^12.3e} | {time_sketch:^12.4f}")
print("-" * 80)


#El modelo Sketching es claramente superior al modelo normal en términos de precisión, eficiencia y rendimiento general. Aunque el r^2 del modelo Sketching
# aún no alcanza valores positivos, su rendimiento es miles de veces mejor que el del modelo normal. Es recomendable adoptar el modelo Sketching para esta tarea,
# con posibles ajustes adicionales para seguir mejorando su precisión, como elevar el grado del polinomio o hacerlo por steps.


### 3. Optimización y Propuesta de Solución

# Basándose en los resultados del modelo, que indican una fuerte relación entre el tráfico y factores como el clima y días feriados,
# se pueden implementar las siguientes soluciones:

# 1) Implementar semáforos adaptativos que ajusten sus ciclos en tiempo real según el flujo vehicular, condiciones climáticas y eventos especiales.
# 2) Desarrollar o integrar aplicaciones móviles que sugieran rutas alternativas basadas en predicciones de tráfico y factores como lluvias o baja visibilidad.
# 3) Mejorar la infraestructura y frecuencia del transporte público, especialmente en días feriados o con condiciones climáticas adversas, para incentivar su uso.
# 4) Trabajar con empresas e instituciones para implementar horarios escalonados y flexibles, reduciendo la cantidad de vehículos en las horas pico.
# 5) En casos extremos donde haya gran volumen de tráfico la apertura de carriles reversibles o
#  en casos de extrema necesidad la reducción de acera para la construcción de un carril auxiliar.