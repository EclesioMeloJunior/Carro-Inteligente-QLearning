# Otimizando fluxos de armazens com Q_Learning algorithm

#importando bibliotecas
import numpy as np

#Configurando parametros gamma e alpha para o Q_Learning
gamma = 0.75
alpha = 0.90

location_to_state = {'A': 0, 
                     'B': 1, 
                     'C': 2,
                     'D': 3,
                     'E': 4,
                     'F': 5, 
                     'G': 6,
                     'H': 7,
                     'I': 8,
                     'J': 9,
                     'K': 10,
                     'L': 11}

state_to_location = {state:location for location, state in location_to_state.items()}

actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#              A  B  C  D  E  F  G  H  I  J  K  L
R = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]])


def calc_route(starting_location, ending_location, pass_before = None):
    R_new = np.copy(R)
    ending_state = location_to_state[ending_location]
    
    R_new[ending_state, ending_state] = 1000  
    
    Q = np.array(np.zeros((12, 12)))
    
    for i in range(1000):
        current_state = np.random.randint(0, 12)
        playable_actions = []
        
        for j in range(12):
            if R_new[current_state, j] > 0:
                playable_actions.append(j)
                
        next_state = np.random.choice(playable_actions)
        
        TD = R_new[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD
    
    route = [starting_location]
    next_location = starting_location
    
    while (next_location != ending_location):
        current_state = location_to_state[next_location]
        next_state = np.argmax(Q[current_state])
        next_location = state_to_location[next_state]
        route.append(next_location)

    return route

print('Rota:')
print(calc_route('J', 'G'))


def best_route(start_location, intermediary_location, ending_location):
    first_route = calc_route(start_location, intermediary_location)
    second_route = calc_route(intermediary_location, ending_location)[1:]
    return first_route + second_route

print(best_route('J', 'A', 'D'))

















