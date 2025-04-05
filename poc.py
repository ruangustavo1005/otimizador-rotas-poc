import os
from dotenv import load_dotenv
import googlemaps
from datetime import datetime, time, timedelta
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from typing import List, Dict, Tuple
from sklearn.cluster import DBSCAN

load_dotenv()


class Paciente:
    def __init__(
        self,
        nome: str,
        hora_consulta: time,
        cidade: str,
        coordenadas: Tuple[float, float],
        acompanhantes: int = 0,
        sensivel: bool = False,
    ):
        self.nome = nome
        self.hora_consulta = datetime.combine(datetime.now(), hora_consulta)
        self.cidade = cidade
        self.coordenadas = coordenadas
        self.acompanhantes = acompanhantes
        self.sensivel = sensivel
        self.total_passageiros = 1 + acompanhantes

    def __str__(self):
        return f"[{self.cidade}] {self.nome} - {self.hora_consulta.strftime('%H:%M')} {'(sensível)' if self.sensivel else ''} + {self.acompanhantes} acomp."


class Veiculo:
    def __init__(self, nome: str, tipo: str, capacidade: int):
        self.nome = nome
        self.tipo = tipo
        self.capacidade = capacidade

    def __str__(self):
        return f"{self.nome} ({self.tipo})"


class Motorista:
    def __init__(self, nome: str, plantao: bool = False):
        self.nome = nome
        self.plantao = plantao

    def __str__(self):
        return f"{self.nome} ({'Plantao' if self.plantao else 'Não Plantao'})"


class OtimizadorRotas:
    def __init__(self, api_key: str):
        self.gmaps = googlemaps.Client(key=api_key)
        self.veiculos = []
        self.motoristas = []
        self.pacientes = []
        self.matriz_distancias = None
        self.matriz_tempos = None
        self.origem = (
            -26.825413199174164,
            -49.27287128764243,
        )  # Coordenadas de José Boiteux

    def adicionar_veiculo(self, veiculo: Veiculo):
        self.veiculos.append(veiculo)

    def adicionar_motorista(self, motorista: Motorista):
        self.motoristas.append(motorista)

    def adicionar_paciente(self, paciente: Paciente):
        self.pacientes.append(paciente)

    def calcular_matriz_distancias_tempos(self):
        """Calcula a matriz de distâncias e tempos entre todos os pontos"""

        self.__log("Calculando matriz de distâncias e tempos...")

        # Lista de todos os pontos (origem + destinos)
        pontos = [self.origem] + [p.coordenadas for p in self.pacientes]
        num_pontos = len(pontos)

        # Inicializa matrizes
        self.matriz_distancias = np.zeros((num_pontos, num_pontos))
        self.matriz_tempos = np.zeros((num_pontos, num_pontos))

        # Calcula distâncias e tempos para cada par de pontos
        for i in range(num_pontos):
            for j in range(i + 1, num_pontos):
                if i != j:
                    # Calcula usando a API do Google Maps
                    resultado = self.gmaps.distance_matrix(
                        origins=[f"{pontos[i][0]},{pontos[i][1]}"],
                        destinations=[f"{pontos[j][0]},{pontos[j][1]}"],
                        mode="driving",
                        language="pt-BR",
                    )

                    try:
                        distancia = resultado["rows"][0]["elements"][0]["distance"][
                            "value"
                        ]  # em metros
                        duracao = resultado["rows"][0]["elements"][0]["duration"][
                            "value"
                        ]  # em segundos
                    except KeyError:
                        # Fallback para distância euclidiana
                        distancia = (
                            self.distancia_euclidiana(pontos[i], pontos[j]) * 100000
                        )  # Conversão aproximada
                        duracao = (
                            distancia / 20
                        )  # Velocidade média de 20 m/s (72 km/h)

                    self.matriz_distancias[i, j] = distancia
                    self.matriz_distancias[j, i] = distancia
                    self.matriz_tempos[i, j] = duracao
                    self.matriz_tempos[j, i] = duracao

        self.__log("Matriz de distâncias e tempos calculada com sucesso!")

    def distancia_euclidiana(self, p1, p2):
        """Calcula a distância euclidiana entre dois pontos geográficos"""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def agrupar_pacientes_por_espaco_tempo(self):
        """Agrupa pacientes usando DBSCAN em duas dimensões: espaço e tempo"""

        self.__log("Agrupando pacientes por espaço-tempo...")

        if not self.matriz_distancias.any() or not self.matriz_tempos.any():
            self.calcular_matriz_distancias_tempos()

        # Preparar dados para clustering
        # Usamos coordenadas normalizadas e hora de consulta como features
        features = []
        for paciente in self.pacientes:
            # Normaliza coordenadas (feature scaling) para equilibrá-las com o tempo
            lat_norm = (
                paciente.coordenadas[0] + 27
            ) * 10  # Normaliza para range aproximado
            lon_norm = (
                paciente.coordenadas[1] + 49
            ) * 10  # Normaliza para range aproximado

            # Converte hora para minutos desde meia-noite
            hora_minutos = (
                paciente.hora_consulta.hour * 60 + paciente.hora_consulta.minute
            )
            hora_norm = hora_minutos / 30  # Normaliza para dar mais peso ao tempo

            features.append([lat_norm, lon_norm, hora_norm])

        X = np.array(features)

        # Executa DBSCAN
        # eps: distância máxima entre pontos no mesmo cluster
        # min_samples: número mínimo de pontos para formar um cluster
        dbscan = DBSCAN(eps=3.0, min_samples=1)
        clusters = dbscan.fit_predict(X)

        # Organiza pacientes por cluster
        pacientes_por_cluster = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in pacientes_por_cluster:
                pacientes_por_cluster[cluster_id] = []
            pacientes_por_cluster[cluster_id].append(self.pacientes[i])

        # Separa pacientes sensíveis em seus próprios clusters
        clusters_finais = []
        for cluster_id, pacientes in pacientes_por_cluster.items():
            # Identifica pacientes sensíveis no cluster
            sensiveis = [p for p in pacientes if p.sensivel]
            normais = [p for p in pacientes if not p.sensivel]

            # Adiciona pacientes sensíveis em clusters individuais
            for paciente in sensiveis:
                clusters_finais.append([paciente])

            # Adiciona pacientes normais se houver
            if normais:
                clusters_finais.append(normais)

        self.__log("Pacientes agrupados por espaço-tempo com sucesso!")

        return clusters_finais

    def resolver_vrptw_para_cluster(self, cluster_pacientes):
        """Resolve o problema de roteamento de veículos com janelas de tempo para um cluster"""

        self.__log("Resolvendo problema de roteamento de veículos com janelas de tempo para um cluster...")

        if not self.matriz_distancias.any() or not self.matriz_tempos.any():
            self.calcular_matriz_distancias_tempos()

        # Índices dos pacientes no cluster
        indices_pacientes = []
        for paciente in cluster_pacientes:
            indices_pacientes.append(
                self.pacientes.index(paciente) + 1
            )  # +1 porque origem é índice 0

        num_locais = len(indices_pacientes) + 1  # +1 para a origem

        # Extrai submatriz de tempos para os pontos relevantes
        indices_completos = [0] + indices_pacientes  # Inclui origem (índice 0)
        tempos_submatriz = np.zeros((num_locais, num_locais))
        for i in range(num_locais):
            for j in range(num_locais):
                tempos_submatriz[i, j] = self.matriz_tempos[
                    indices_completos[i], indices_completos[j]
                ]

        # Cria o modelo de roteamento
        num_veiculos = min(len(cluster_pacientes), len(self.veiculos))
        manager = pywrapcp.RoutingIndexManager(num_locais, num_veiculos, 0)
        routing = pywrapcp.RoutingModel(manager)

        # Define a função de custo (tempo de viagem)
        def tempo_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(tempos_submatriz[from_node, to_node])

        transit_callback_index = routing.RegisterTransitCallback(tempo_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Adiciona restrições de capacidade
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            if from_node == 0:  # Origem
                return 0
            paciente_idx = indices_pacientes[from_node - 1]
            return self.pacientes[paciente_idx - 1].total_passageiros

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [
                v.capacidade for v in self.veiculos[:num_veiculos]
            ],  # vehicle maximum capacities
            True,  # start cumul to zero
            "Capacity",
        )

        # Adiciona janelas de tempo
        time_dimension_name = "Time"
        routing.AddDimension(
            transit_callback_index,
            60 * 60,  # permite espera de até 1 hora
            24 * 60 * 60,  # limite diário de 24 horas
            False,  # não inicia em zero
            time_dimension_name,
        )
        time_dimension = routing.GetDimensionOrDie(time_dimension_name)

        # Adiciona janelas de tempo para cada paciente
        for i, paciente_idx in enumerate(indices_pacientes):
            index = manager.NodeToIndex(i + 1)
            paciente = self.pacientes[paciente_idx - 1]

            # Converte hora da consulta para segundos desde meia-noite
            consulta_segundos = (
                paciente.hora_consulta.hour * 3600 + paciente.hora_consulta.minute * 60
            )

            # Define janela de tempo: deve chegar pelo menos 15 minutos antes
            # e no máximo 2 horas antes da consulta
            earliest_arrival = consulta_segundos - 2 * 60 * 60  # 2 horas antes
            latest_arrival = consulta_segundos - 15 * 60  # 15 minutos antes

            # Garante que não seja negativo
            earliest_arrival = max(0, earliest_arrival)

            time_dimension.CumulVar(index).SetRange(earliest_arrival, latest_arrival)

        # Configura a estratégia de busca
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 30  # limite de tempo para resolução

        # Resolve o problema
        solution = routing.SolveWithParameters(search_parameters)

        rotas = []
        if solution:
            # Processa a solução
            for vehicle_id in range(num_veiculos):
                index = routing.Start(vehicle_id)
                if not routing.IsEnd(
                    solution.Value(routing.NextVar(index))
                ):  # Se o veículo é usado
                    rota = []
                    total_passageiros = 0

                    while not routing.IsEnd(index):
                        node_index = manager.IndexToNode(index)
                        if node_index > 0:  # Não é a origem
                            paciente_idx = indices_pacientes[node_index - 1]
                            paciente = self.pacientes[paciente_idx - 1]
                            rota.append(paciente)
                            total_passageiros += paciente.total_passageiros
                        index = solution.Value(routing.NextVar(index))

                    if rota:  # Se a rota não estiver vazia
                        # Calculando horário de saída baseado no paciente que tem
                        # o horário de consulta mais cedo na rota
                        primeiro_paciente = min(rota, key=lambda p: p.hora_consulta)
                        tempo_viagem = self.matriz_tempos[
                            0, self.pacientes.index(primeiro_paciente) + 1
                        ]

                        # Adiciona tempo para paradas anteriores
                        if len(rota) > 1:
                            tempo_viagem += (
                                (len(rota) - 1) * 15 * 60
                            )  # 15 minutos por parada adicional

                        hora_saida = primeiro_paciente.hora_consulta - timedelta(
                            seconds=int(tempo_viagem + 30 * 60)
                        )  # 30 min de margem

                        # Calculando horário de retorno baseado no paciente que tem
                        # o horário de consulta mais tarde na rota
                        ultimo_paciente = max(rota, key=lambda p: p.hora_consulta)
                        tempo_retorno = self.matriz_tempos[
                            self.pacientes.index(ultimo_paciente) + 1, 0
                        ]

                        # Adiciona tempo de consulta (1h) e espera
                        hora_retorno = ultimo_paciente.hora_consulta + timedelta(
                            hours=1, seconds=int(tempo_retorno)
                        )

                        rotas.append(
                            {
                                "veiculo": self.veiculos[vehicle_id].nome,
                                "pacientes": rota,
                                "hora_saida": hora_saida,
                                "hora_retorno": hora_retorno,
                                "total_passageiros": total_passageiros,
                            }
                        )

        self.__log("Problema de roteamento de veículos com janelas de tempo resolvido com sucesso!")

        return rotas

    def otimizar_rotas(self) -> List[Dict]:
        """Otimiza as rotas de forma inteligente, agrupando por espaço-tempo e resolvendo VRPTW"""

        self.__log("Iniciando otimização de rotas...")

        # Calcula matriz de distâncias e tempos
        self.calcular_matriz_distancias_tempos()

        # Agrupa pacientes por espaço-tempo
        clusters = self.agrupar_pacientes_por_espaco_tempo()

        # Resolve VRPTW para cada cluster
        todas_rotas = []
        for cluster in clusters:
            rotas = self.resolver_vrptw_para_cluster(cluster)
            todas_rotas.extend(rotas)

        # Ordena rotas por hora de saída
        todas_rotas.sort(key=lambda r: r["hora_saida"])

        # Atribui motoristas e veículos às rotas, verificando disponibilidade
        rotas_finais = []
        motoristas_disponiveis = [m for m in self.motoristas if not m.plantao]
        veiculos_agenda = {
            v.nome: [] for v in self.veiculos
        }  # Registra uso de cada veículo
        motorista_atual = 0

        for rota in todas_rotas:
            # Atribui um motorista disponível
            motorista = motoristas_disponiveis[
                motorista_atual % len(motoristas_disponiveis)
            ]
            motorista_atual += 1

            # Encontra um veículo disponível do mesmo tipo
            veiculo_original = rota["veiculo"]
            tipo_veiculo = next(
                (v.tipo for v in self.veiculos if v.nome == veiculo_original), "carro"
            )
            capacidade_necessaria = rota["total_passageiros"]

            veiculo_disponivel = None
            for veiculo in self.veiculos:
                # Verifica se o veículo tem capacidade suficiente
                if veiculo.capacidade < capacidade_necessaria:
                    continue

                # Verifica se o veículo está livre no período necessário
                horario_conflito = False
                for uso in veiculos_agenda.get(veiculo.nome, []):
                    # Verifica sobreposição de horários
                    if not (
                        rota["hora_retorno"] <= uso["inicio"]
                        or rota["hora_saida"] >= uso["fim"]
                    ):
                        horario_conflito = True
                        break

                if not horario_conflito:
                    veiculo_disponivel = veiculo
                    # Registra o uso deste veículo
                    veiculos_agenda[veiculo.nome].append(
                        {"inicio": rota["hora_saida"], "fim": rota["hora_retorno"]}
                    )
                    break

            # Se não encontrou veículo disponível, pula esta rota
            if not veiculo_disponivel:
                print(
                    f"AVISO: Não foi possível alocar veículo para rota com pacientes: {[p.nome for p in rota['pacientes']]}"
                )
                continue

            rotas_finais.append(
                {
                    "motorista": motorista,
                    "veiculo": veiculo_disponivel,
                    "pacientes": [p for p in rota["pacientes"]],
                    "hora_saida": rota["hora_saida"].strftime("%H:%M"),
                    "hora_retorno": rota["hora_retorno"].strftime("%H:%M"),
                    "total_passageiros": rota["total_passageiros"],
                }
            )

        self.__log("Otimização de rotas concluída com sucesso!")

        return rotas_finais

    def __log(self, message: str):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


# Exemplo de uso
if __name__ == "__main__":
    # Substitua pela sua chave API do Google Maps
    API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

    instante_inicial = datetime.now()

    otimizador = OtimizadorRotas(API_KEY)

    # Adicionar veículos
    for i in range(7):  # 7 carros
        otimizador.adicionar_veiculo(Veiculo(f"Carro {i}", "carro", 4))
    for i in range(2):  # 2 vans
        otimizador.adicionar_veiculo(Veiculo(f"Van {i}", "van", 15))

    # Adicionar motoristas
    for i in range(8):
        otimizador.adicionar_motorista(Motorista(f"Motorista {i+1}", i == 0))

    # Adicionar pacientes manualmente
    for paciente in [
        Paciente("anita", time(7, 00), "Timbó", (-26.82541, -49.27287), 1),
        Paciente("marcelo", time(6, 00), "Blumenau", (-26.92672, -49.05587)),
        Paciente("iria", time(7, 00), "Rio do Sul", (-27.21920, -49.64346), 1),
        Paciente("larissa", time(9, 00), "Rio do Sul", (-27.21920, -49.64346), 2),
        Paciente("marlene", time(8, 30), "Rio do Sul", (-27.21426, -49.64511), 1),
        Paciente("olivino", time(9, 30), "Rio do Sul", (-27.22142, -49.64246), 1),
        Paciente("rogerio", time(9, 00), "Rio do Sul", (-27.21619, -49.64262)),
        Paciente("vanderlei", time(11, 00), "Presidente Getúlio", (-27.05160, -49.62177)),
        Paciente("edielson", time(11, 00), "Brusque", (-27.09575, -48.91845)),
        Paciente("maria", time(12, 00), "Brusque", (-27.09575, -48.91845), 1),
        Paciente("paulo", time(12, 00), "Brusque", (-27.09575, -48.91845)),
        Paciente("sebastiao", time(8, 30), "Ibirama", (-27.06163, -49.51926), 1),
        Paciente("silvana", time(11, 00), "Ibirama", (-27.06163, -49.51926), 1),
        Paciente("etelvina", time(10, 00), "Rio do Sul", (-27.21891, -49.64395), 1),
        Paciente("jose", time(10, 00), "Ibirama", (-27.06163, -49.51926), 1),
        Paciente("gilberto", time(15, 00), "Timbó", (-26.82919, -49.27167)),
        Paciente("mariana", time(14, 00), "Blumenau", (-26.91635, -49.05817), 1),
        Paciente("andrei", time(14, 00), "Pomerode", (-26.71633, -49.16787), 1, True),
        Paciente("jairo", time(14, 00), "Presidente Getúlio", (-27.04627, -49.61920), 0, True),
    ]:
        otimizador.adicionar_paciente(paciente)

    # Otimizar rotas
    rotas = otimizador.otimizar_rotas()

    # Imprimir resultados
    print("\nRotas otimizadas:")
    print("=" * 50)
    for rota in rotas:
        print(f"\nMotorista: {rota['motorista']}")
        print(f"Veículo: {rota['veiculo']}")
        print(f"Pacientes:")
        for paciente in rota['pacientes']:
            print(f"  - {paciente}")
        print(f"Hora de saída: {rota['hora_saida']}")
        print(f"Hora de retorno: {rota['hora_retorno']}")
        print(f"Total de passageiros: {rota['total_passageiros']}")
        print("-" * 50)

    instante_final = datetime.now()
    print(f"\nTempo de execução: {(instante_final - instante_inicial).total_seconds()} segundos")
