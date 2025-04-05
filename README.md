# Otimizador de Rotas - Transporte de Pacientes

Sistema de otimização para transporte de pacientes da secretaria de saúde, que gera roteiros eficientes para motoristas e veículos, considerando:

- Horários de consultas médicas
- Localização geográfica dos pacientes
- Capacidade dos veículos
- Disponibilidade de motoristas
- Pacientes com necessidades especiais
- Número de acompanhantes

## Algoritmos Utilizados

- **Google Maps API**: Cálculo real de tempos e distâncias
- **DBSCAN (Clustering)**: Agrupamento de pacientes por proximidade e horário
- **OR-Tools (VRPTW)**: Solução de roteamento com janelas de tempo

## Requisitos

- Python 3.6+
- Pacotes: googlemaps, numpy, ortools, scikit-learn (presentes em `requirements.txt`)
- Chave de API do Google Maps (para cálculos de distância/tempo)

## Execução

Crie um `.env` baseado em `.env.example`, preencha o mesmo e execute:

```bash
python poc.py
```

O sistema produzirá roteiros otimizados de transporte, garantindo que os pacientes cheguem às consultas no horário correto, usando eficientemente os recursos disponíveis de veículos e motoristas. 