import json
import joblib
import numpy as np

# Inicialização do modelo
def init():
    global model
    # Carrega o modelo a partir do arquivo
    model_path = "modelo/modelo.pkl"  # Caminho relativo ao modelo
    model = joblib.load(model_path)

# Função de previsão
def run(data):
    try:
        # Converte os dados de entrada em formato JSON
        input_data = json.loads(data)
        
        # Garante que os dados estejam em formato adequado (ex.: lista de listas)
        features = np.array(input_data["features"])
        
        # Realiza a previsão
        predictions = model.predict(features)
        
        # Retorna as previsões em formato JSON
        return json.dumps({"predictions": predictions.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
