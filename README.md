# Modelo de Previsão com Azure Machine Learning

Este projeto demonstra o desenvolvimento e a implantação de um modelo de previsão utilizando o Azure Machine Learning. O objetivo é criar um ponto de extremidade configurado para realizar inferências em tempo real e disponibilizá-lo como parte de um portfólio profissional.

Descrição do Projeto
O projeto inclui as etapas de:

Preparação e pré-processamento dos dados.
Treinamento do modelo de aprendizado de máquina.
Registro do modelo no Azure ML.
Configuração e implantação de um endpoint para inferências em tempo real.
Documentação do processo e disponibilização dos arquivos necessários no repositório.
Arquivos no Repositório
README.md: Documentação detalhada do projeto e do passo a passo.
endpoint-config.json: Configuração do ponto de extremidade do modelo.
score.py: Script usado para processar as requisições enviadas ao endpoint.
environment.yml: Definição do ambiente de execução com as dependências necessárias.

Passo 1: Criar um novo repositório no GitHub
Acesse o GitHub e crie um novo repositório com um nome que reflita o objetivo do projeto, como previsao-modelo-azure ou algo similar.
Inicialize o repositório com um arquivo README.md.
Passo 2: Configurar o ambiente no Azure ML
Criação do Workspace:

Acesse o portal do Azure e procure por Azure Machine Learning.
Crie um novo workspace (espaço de trabalho).
Escolha a região, grupo de recursos e configure os detalhes básicos.
Preparação dos Dados:

Faça upload dos dados de entrada no Azure Blob Storage ou diretamente no Data Assets do Azure ML Studio.
Se necessário, use notebooks ou scripts no Azure para pré-processar os dados.
Criação e Treinamento do Modelo:

No Azure ML Studio, crie um Compute Cluster para rodar o treinamento do modelo.
Use a interface do Designer ou notebooks no workspace para treinar o modelo.
Escolha algoritmos adequados (como regressão, classificação ou outros) dependendo do problema.
Registro do Modelo:

Após o treinamento, registre o modelo no workspace.
Passo 3: Configurar o Ponto de Extremidade (Endpoint)
Implantar o Modelo:

No Azure ML Studio, vá para a aba Endpoints.
Configure o Real-time Endpoint:
Escolha o modelo registrado.
Configure as dependências no arquivo score.py e o ambiente do Docker no arquivo environment.yml ou equivalente.
Implante o modelo.
Teste o Endpoint:

Use a ferramenta de teste integrada no Azure ML para enviar requisições de exemplo e verificar as respostas.
Passo 4: Documentar o Processo
Escreva um README.md explicando:

O objetivo do projeto.
O dataset utilizado e o pré-processamento (se aplicável).
Como o modelo foi treinado.
O processo de configuração do ponto de extremidade no Azure ML.
Instruções de como usar o ponto de extremidade (por exemplo, com curl ou uma requisição em Python).
Exporte o arquivo JSON do ponto de extremidade:

No Azure ML Studio, vá para o endpoint criado e faça o download do arquivo .json com a configuração.
Passo 5: Organizar o Repositório
Adicione os seguintes arquivos ao repositório:

README.md: Contendo o passo a passo detalhado.
O arquivo .json do endpoint.
(Opcional) Scripts adicionais, como o score.py ou environment.yml, caso relevantes.
Faça o commit e o push das alterações para o repositório no GitHub.

Passo 6: Compartilhar o Link
Copie o link do repositório e envie conforme instruído no botão "Entregar Projeto".
Exemplo de Estrutura do Repositório:
plaintext
Copiar
Editar
previsao-modelo-azure/
│
├── README.md           # Documentação do processo
├── endpoint-config.json # Arquivo JSON do ponto de extremidade
├── score.py            # (Opcional) Script de inferência
└── environment.yml     # (Opcional) Ambiente do Docker para a implantação


Aqui estão exemplos de como os arquivos `endpoint-config.json`, `score.py` e `environment.yml` podem ser estruturados:

---

### **1. Arquivo `endpoint-config.json`**
Este arquivo descreve a configuração do ponto de extremidade criado no Azure ML.

```json
{
  "name": "modelo-previsao-endpoint",
  "scoringUri": "https://<seu-endpoint>.azurewebsites.net/score",
  "key": "sua-chave-de-acesso",
  "swaggerUri": "https://<seu-endpoint>.azurewebsites.net/swagger.json",
  "description": "Endpoint para inferência do modelo de previsão",
  "computeType": "ACI",
  "environment": {
    "dependencies": ["numpy", "pandas", "scikit-learn"],
    "pythonVersion": "3.8"
  }
}
```

---

### **2. Arquivo `score.py`**
Este script define como o modelo processa as requisições recebidas pelo endpoint.

```python
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
```

---

### **3. Arquivo `environment.yml`**
Este arquivo define as dependências do ambiente Docker para a implantação do modelo.

```yaml
name: modelo-previsao-env
channels:
  - defaults
dependencies:
  - python=3.8
  - pip
  - pip:
      - numpy
      - pandas
      - scikit-learn
      - azureml-defaults
```

---

### **Como esses arquivos funcionam juntos:**
1. O **`endpoint-config.json`** contém informações sobre como acessar o endpoint e o ambiente configurado.
2. O **`score.py`** lida com o processamento de entrada e saída no endpoint.
3. O **`environment.yml`** garante que todas as dependências estejam instaladas no ambiente em que o endpoint será executado.

Se precisar de ajustes específicos para o seu caso de uso, me avise! 😊
