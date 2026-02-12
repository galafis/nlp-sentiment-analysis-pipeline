# 🚀 Nlp Sentiment Analysis Pipeline

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.10-0194E2.svg)](https://mlflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit-learn-1.4-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](#english) | [Português](#português)

---

## English

### 🎯 Overview

**Nlp Sentiment Analysis Pipeline** — Professional Python project

Total source lines: **840** across **12** files in **1** language.

### ✨ Key Features

- **Production-Ready Architecture**: Modular, well-documented, and following best practices
- **Comprehensive Implementation**: Complete solution with all core functionality
- **Clean Code**: Type-safe, well-tested, and maintainable codebase
- **Easy Deployment**: Docker support for quick setup and deployment

### 🚀 Quick Start

#### Prerequisites
- Python 3.12+
- Docker and Docker Compose (optional)

#### Installation

1. **Clone the repository**
```bash
git clone https://github.com/galafis/nlp-sentiment-analysis-pipeline.git
cd nlp-sentiment-analysis-pipeline
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

#### Running

```bash
python src/api/app.py
```

## 🐳 Docker

```bash
# Build the image
docker build -t nlp-sentiment-analysis-pipeline .

# Run the container
docker run -p 8000:8000 nlp-sentiment-analysis-pipeline
```

### 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov --cov-report=html

# Run with verbose output
pytest -v
```

### 📁 Project Structure

```
nlp-sentiment-analysis-pipeline/
├── assets/
├── data/
│   ├── processed/
│   └── raw/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py
│   ├── data/
│   │   └── __init__.py
│   ├── features/
│   │   └── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── sentiment_analyzer.py
│   ├── utils/
│   │   └── __init__.py
│   ├── visualization/
│   │   └── __init__.py
│   └── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_models.py
├── README.md
├── README_OLD.md
├── pytest.ini
├── requirements.txt
└── setup.py
```

### 🛠️ Tech Stack

| Technology | Usage |
|------------|-------|
| Python | 12 files |

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 👤 Author

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-demetrios-lafis)

---

## Português

### 🎯 Visão Geral

**Nlp Sentiment Analysis Pipeline** — Professional Python project

Total de linhas de código: **840** em **12** arquivos em **1** linguagem.

### ✨ Funcionalidades Principais

- **Arquitetura Pronta para Produção**: Modular, bem documentada e seguindo boas práticas
- **Implementação Completa**: Solução completa com todas as funcionalidades principais
- **Código Limpo**: Type-safe, bem testado e manutenível
- **Fácil Implantação**: Suporte Docker para configuração e implantação rápidas

### 🚀 Início Rápido

#### Pré-requisitos
- Python 3.12+
- Docker e Docker Compose (opcional)

#### Instalação

1. **Clone the repository**
```bash
git clone https://github.com/galafis/nlp-sentiment-analysis-pipeline.git
cd nlp-sentiment-analysis-pipeline
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

#### Execução

```bash
python src/api/app.py
```

### 🧪 Testes

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov --cov-report=html

# Run with verbose output
pytest -v
```

### 📁 Estrutura do Projeto

```
nlp-sentiment-analysis-pipeline/
├── assets/
├── data/
│   ├── processed/
│   └── raw/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py
│   ├── data/
│   │   └── __init__.py
│   ├── features/
│   │   └── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── sentiment_analyzer.py
│   ├── utils/
│   │   └── __init__.py
│   ├── visualization/
│   │   └── __init__.py
│   └── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_models.py
├── README.md
├── README_OLD.md
├── pytest.ini
├── requirements.txt
└── setup.py
```

### 🛠️ Stack Tecnológica

| Tecnologia | Uso |
|------------|-----|
| Python | 12 files |

### 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

### 👤 Autor

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-demetrios-lafis)
