
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint

st.set_page_config(page_title="Simulateur de Zones Humides", layout="wide", page_icon="🌿")

class WetlandSimulator:
    def __init__(self):
        # Default parameter values
        self.default_params = {
            'r': 0.5,  # Natural growth rate of fish
            'K': 1000.0,  # Carrying capacity of fish
            'alpha_B': 0.3,  # Impact of predation on fish
            'alpha_T': 0.1,  # Tourist fishing rate
            'gamma_1': 100.0,  # Half-saturation constant for fish-bird interaction
            'gamma_2': 50.0,   # Half-saturation constant for bird-tourist interaction
            'gamma_3': 20.0,   # Half-saturation constant for capital
            'd': 0.1,          # Natural mortality rate of birds
            'beta': 0.4,       # Reproduction rate of birds
            'lambda_bird': 0.01,  # Poaching rate
            'Lambda_Xm': 5.0,     # Amplitude of bird migration variation
            'Lambda_P': 365.0,    # Period of migration
            'Lambda_phi': 0.0,    # Phase shift of migration
            'Lambda_M': 10.0,     # Mean migration rate
            'mu_B': 0.3,          # Attractiveness related to birds
            'mu_C': 0.2,          # Attractiveness of infrastructure
            'alpha': 0.1,         # Competition rate
            'a_Xm': 0.05,         # Amplitude of competition variation
            'a_P': 365.0,         # Period of competition
            'a_phi': 0.0,         # Phase shift of competition
            'a_M': 0.1,           # Mean competition rate
            'Gamma_Xm': 2.0,      # Amplitude of tourist fishing variation
            'Gamma_P': 365.0,     # Period of tourist fishing
            'Gamma_phi': 0.0,     # Phase shift of tourist fishing
            'Gamma_M': 1.0,       # Mean tourist fishing rate
            'delta': 0.05,        # Infrastructure degradation rate
            'epsilon_B': 0.2,     # Investment rate based on birds
            'epsilon_T': 0.3      # Investment rate based on tourists
        }
        
        self.default_initial_conditions = {
            'F_0': 500,   # Initial fish population
            'B_0': 200,   # Initial bird population
            'T_0': 50,    # Initial number of tourists
            'C_0': 1000   # Initial capital
        }

    def system_equations(self, state, t, params):
        F, B, T, C = state
        
        Lambda_t = params['Lambda_Xm'] * np.sin(2 * np.pi * t / params['Lambda_P'] + params['Lambda_phi']) + params['Lambda_M']
        a_t = params['a_Xm'] * np.sin(2 * np.pi * t / params['a_P'] + params['a_phi']) + params['a_M']
        Gamma_t = params['Gamma_Xm'] * np.sin(2 * np.pi * t / params['Gamma_P'] + params['Gamma_phi']) + params['Gamma_M']
        
        dF = params['r'] * F * (1 - F / params['K']) - params['alpha_B'] * F * B / (F + params['gamma_1'])
        dB = (-params['d'] * B + params['beta'] * F * B / (F + params['gamma_1']) + Lambda_t - Gamma_t * B - params['lambda_bird'] * B * T)
        dT = T * (params['mu_B'] * B / (B + params['gamma_2']) + params['mu_C'] * C / (C + params['gamma_3'] * (T + 1)) - params['alpha'] * T - a_t)
        dC = -params['delta'] * C + params['epsilon_B'] * B + params['epsilon_T'] * T 

        


        
        return [dF, dB, dT, dC]

    def simulate(self, params, initial_conditions, t_span):
        t = np.linspace(0, t_span, int(t_span))
        initial_state = [initial_conditions['F_0'], initial_conditions['B_0'], initial_conditions['T_0'], initial_conditions['C_0']]
        
        try:
            solution = odeint(self.system_equations, initial_state, t, args=(params,))
            return pd.DataFrame({
                'time': t,
                'Fish': solution[:, 0],
                'Birds': solution[:, 1],
                'Tourists': solution[:, 2],
                'Capital': solution[:, 3]
            })
        except Exception as e:
            st.error(f"Erreur lors de la simulation: {str(e)}")
            return None

def create_plotly_figures(results):
    """Create Plotly visualizations"""
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Population de poissons", "Population d'oiseaux", "Nombre de touristes", "Capital"),
                        vertical_spacing=0.15,  # Augmenter l'espacement vertical
                        horizontal_spacing=0.1)  # Augmenter l'espacement horizontal

    colors = {
        'Fish': 'rgba(30, 144, 255, 0.3)',    # rgba for dodger blue with transparency
        'Birds': 'rgba(46, 139, 87, 0.3)',    # rgba for sea green with transparency
        'Tourists': 'rgba(255, 99, 71, 0.3)', # rgba for tomato with transparency
        'Capital': 'rgba(147, 112, 219, 0.3)' # rgba for medium purple with transparency
    }
    
    for column, color in colors.items():
        fig.add_trace(go.Scatter(
            x=results['time'],
            y=results[column],
            name=column,
            line=dict(color=color),
            fill='tozeroy'
        ), row=1 if column in ['Fish', 'Birds'] else 2,
           col=1 if column in ['Fish', 'Tourists'] else 2)
    
    fig.update_layout(height=700,
                      title_text="Dynamique du système de zones humides",
                      title_font_size=16,
                      showlegend=True,
                      template='plotly_white')
    fig.update_xaxes(title_text="Temps (jours)")
    fig.update_yaxes(title_text="Population / Valeur")
    
    return fig

def main():
    # En-tête avec logo, informations sur l'université et les intervenants
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("logo_unchk.png", width=350)  # Ajouter le logo en haut à gauche

    with col2:
        st.markdown("""
        ### Article : Modèle mathématique pour la durabilité du tourisme dans les zones humides
        **Modélisation Mathématique** : Dr. Oumar Diop (oumar.diop@unchk.edu.sn)  
        **Modélisation Mathématique** : Dr. Abdou Séne (abdou.sene@unchk.sn)   
        """)

    st.title("🌿 Simulateur de Dynamique des Zones Humides")
    st.markdown("""
    ### Exploration des interactions écologiques et touristiques
    Ce simulateur avancé modélise les interactions complexes entre :
    - Population de poissons
    - Population d'oiseaux
    - Afflux de touristes
    - Développement économique (capital)
    
    Utilisez les paramètres à gauche pour explorer différents scénarios.
    """)
    
    simulator = WetlandSimulator()
    
    with st.sidebar:
        st.header("🛠 Paramètres de Simulation")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Écologie", "Tourisme", "Fonctions Périodiques", "Conditions Initiales"])
        
        with tab1:
            st.subheader("Paramètres Écologiques")
            params = {
                'r': st.slider("Taux de croissance des poissons", min_value=0.1, max_value=1.0, value=simulator.default_params['r'], step=0.01),
                'K': st.slider("Capacité de charge des poissons", min_value=100, max_value=2000, value=int(simulator.default_params['K']), step=100),
                'alpha_B': st.slider("Impact de prédation", min_value=0.1, max_value=1.0, value=simulator.default_params['alpha_B'], step=0.01),
                'd': st.slider("Mortalité naturelle des oiseaux", min_value=0.05, max_value=1.00, value=simulator.default_params['d'], step=0.01),
                'beta': st.slider("Taux de reproduction des oiseaux", min_value=0.1, max_value=1.0, value=simulator.default_params['beta'], step=0.01)
            }
        
        with tab2:
            st.subheader("Paramètres Touristiques")
            params.update({
                'mu_B': st.slider("Attractivité des oiseaux", min_value=0.1, max_value=1.0, value=simulator.default_params['mu_B'], step=0.01),
                'mu_C': st.slider("Attractivité des infrastructures", min_value=0.1, max_value=1.0, value=simulator.default_params['mu_C'], step=0.01),
                'alpha_T': st.slider("Taux de pêche des touristes", min_value=0.0, max_value=1.0, value=simulator.default_params['alpha_T'], step=0.01)
            })
        
        with tab3:
            st.subheader("Fonctions Périodiques")
            params.update({
                'Lambda_Xm': st.slider("Amplitude de la migration des oiseaux", min_value=0.0, max_value=10.0, value=simulator.default_params['Lambda_Xm'], step=0.5),
                'Lambda_P': st.slider("Période de la migration des oiseaux", min_value=10, max_value=3650, value=int(simulator.default_params['Lambda_P']), step=10),
                'Lambda_phi': st.slider("Déphasage de la migration des oiseaux", min_value=0.0, max_value=2 * np.pi, value=simulator.default_params['Lambda_phi'], step=0.1),
                'Lambda_M': st.slider("Moyenne de la migration des oiseaux", min_value=0.0, max_value=20.0, value=simulator.default_params['Lambda_M'], step=0.5),

                'a_Xm': st.slider("Amplitude de la variation de compétition", min_value=0.0, max_value=1.0, value=simulator.default_params['a_Xm'], step=0.01),
                'a_P': st.slider("Période de la compétition", min_value=10, max_value=3650, value=int(simulator.default_params['a_P']), step=10),
                'a_phi': st.slider("Déphasage de la compétition", min_value=0.0, max_value=2 * np.pi, value=simulator.default_params['a_phi'], step=0.1),
                'a_M': st.slider("Moyenne de la compétition", min_value=0.0, max_value=1.0, value=simulator.default_params['a_M'], step=0.01),

                'Gamma_Xm': st.slider("Amplitude de la pêche touristique", min_value=0.0, max_value=5.0, value=simulator.default_params['Gamma_Xm'], step=0.5),
                'Gamma_P': st.slider("Période de la pêche touristique", min_value=10, max_value=3650, value=int(simulator.default_params['Gamma_P']), step=10),
                'Gamma_phi': st.slider("Déphasage de la pêche touristique", min_value=0.0, max_value=2 * np.pi, value=simulator.default_params['Gamma_phi'], step=0.1),
                'Gamma_M': st.slider("Moyenne de la pêche touristique", min_value=0.0, max_value=5.0, value=simulator.default_params['Gamma_M'], step=0.5)
            })
        
        with tab4:
            st.subheader("Conditions Initiales")
            initial_conditions = {
                'F_0': st.number_input("Population initiale de poissons", min_value=10, max_value=10000, value=simulator.default_initial_conditions['F_0']),
                'B_0': st.number_input("Population initiale d'oiseaux", min_value=10, max_value=500, value=simulator.default_initial_conditions['B_0']),
                'T_0': st.number_input("Nombre initial de touristes", min_value=10, max_value=200, value=simulator.default_initial_conditions['T_0']),
                'C_0': st.number_input("Capital initial", min_value=100, max_value=50000, value=simulator.default_initial_conditions['C_0'])
            }
        
        t_span = st.slider("Durée de simulation (jours)", min_value=100, max_value=36500, value=365)

        # Update params with the rest of the default parameters
        params.update({k: v for k, v in simulator.default_params.items() if k not in params})

    results = simulator.simulate(params, initial_conditions, t_span)

    if results is not None:
         fig = create_plotly_figures(results)
         st.subheader("🔍 Résultats de Simulation")
         st.plotly_chart(fig, use_container_width=True)

         col1, col2 = st.columns(2)
         with col1:
             st.metric("Population finale de poissons", f"{results['Fish'].iloc[-1]:,.0f}")
             st.metric("Population finale d'oiseaux", f"{results['Birds'].iloc[-1]:,.0f}")
         with col2:
             st.metric("Nombre final de touristes", f"{results['Tourists'].iloc[-1]:,.0f}")
             st.metric("Capital final", f"{results['Capital'].iloc[-1]:,.0f}")

if __name__ == "__main__":
    main()

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@___(((((((((((((((()))))))))))))))___@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# import streamlit as st
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from scipy.integrate import odeint

# st.set_page_config(page_title="Simulateur de Zones Humides", layout="wide", page_icon="🌿")

# class WetlandSimulator:
#     def __init__(self):
#         # Valeurs par défaut des paramètres
#         self.default_params = {
#             'r': 0.5,
#             'K': 1000.0,
#             'alpha_B': 0.3,
#             'alpha_T': 0.1,
#             'gamma_1': 100.0,
#             'gamma_2': 50.0,
#             'gamma_3': 20.0,
#             'd': 0.1,
#             'beta': 0.4,
#             'lambda_bird': 0.01,
#             'Lambda_Xm': 5.0,
#             'Lambda_P': 365.0,
#             'Lambda_phi': 0.0,
#             'Lambda_M': 10.0,
#             'mu_B': 0.3,
#             'mu_C': 0.2,
#             'alpha': 0.1,
#             'a_Xm': 0.05,
#             'a_P': 365.0,
#             'a_phi': 0.0,
#             'a_M': 0.1,
#             'Gamma_Xm': 2.0,
#             'Gamma_P': 365.0,
#             'Gamma_phi': 0.0,
#             'Gamma_M': 1.0,
#             'delta': 0.05,
#             'epsilon_B': 0.2,
#             'epsilon_T': 0.3
#         }
        
#         self.default_initial_conditions = {
#             'F_0': 500, 
#             'B_0': 200, 
#             'T_0': 50, 
#             'C_0': 1000 
#         }

#     def system_equations(self, state, t, params):
#         F, B, T, C = state
        
#         # Calcul des termes périodiques pour les interactions
#         Lambda_t = params['Lambda_Xm'] * np.sin(2 * np.pi * t / params['Lambda_P'] + params['Lambda_phi']) + params['Lambda_M']
#         a_t = params['a_Xm'] * np.sin(2 * np.pi * t / params['a_P'] + params['a_phi']) + params['a_M']
#         Gamma_t = params['Gamma_Xm'] * np.sin(2 * np.pi * t / params['Gamma_P'] + params['Gamma_phi']) + params['Gamma_M']
        
#         # Équations différentielles
#         dF = params['r'] * F * (1 - F / params['K']) - params['alpha_B'] * F * B / (F + params['gamma_1'])
#         dB = (-params['d'] * B + params['beta'] * F * B / (F + params['gamma_1']) + Lambda_t - Gamma_t * B - params['lambda_bird'] * B * T)
#         dT = T * (params['mu_B'] * B / (B + params['gamma_2']) + params['mu_C'] * C / (C + params['gamma_3'] * (T + 1)) - params['alpha'] * T - a_t)
#         dC = -params['delta'] * C + params['epsilon_B'] * B + params['epsilon_T'] * T
        
#         return [dF, dB, dT, dC]

#     def simulate(self, params, initial_conditions, t_span):
#         t = np.linspace(0, t_span, int(t_span))
#         initial_state = [initial_conditions['F_0'], initial_conditions['B_0'], initial_conditions['T_0'], initial_conditions['C_0']]
        
#         try:
#             solution = odeint(self.system_equations, initial_state, t, args=(params,))
#             return pd.DataFrame({
#                 'time': t,
#                 'Fish': solution[:, 0],
#                 'Birds': solution[:, 1],
#                 'Tourists': solution[:, 2],
#                 'Capital': solution[:, 3]
#             })
#         except Exception as e:
#             st.error(f"Erreur lors de la simulation: {str(e)}")
#             return None

# def create_plotly_figures(results):
#     """Créer des visualisations avec Plotly"""
#     fig = make_subplots(rows=2, cols=2, subplot_titles=("Population de poissons", "Population d'oiseaux", "Nombre de touristes", "Capital"), vertical_spacing=0.15, horizontal_spacing=0.1)

#     colors = {
#         'Fish': 'rgba(30, 144, 255, 0.3)',
#         'Birds': 'rgba(46, 139, 87, 0.3)',
#         'Tourists': 'rgba(255, 99, 71, 0.3)',
#         'Capital': 'rgba(147, 112, 219, 0.3)'
#     }

#     for column, color in colors.items():
#         fig.add_trace(go.Scatter(
#             x=results['time'], 
#             y=results[column], 
#             name=column, 
#             line=dict(color=color), 
#             fill='tozeroy'
#         ), row=1 if column in ['Fish', 'Birds'] else 2, col=1 if column in ['Fish', 'Tourists'] else 2)

#     fig.update_layout(height=700, title_text="Dynamique du système de zones humides", title_font_size=16)
#     fig.update_xaxes(title_text="Temps (jours)")
#     fig.update_yaxes(title_text="Population / Valeur")
    
#     return fig

# def main():
#     # En-tête avec logo et informations sur l'université
#     col1, col2 = st.columns([1, 3])
#     with col1:
#         st.image("logo_unchk.png", width=350)
    
#     with col2:
#         st.markdown(""" 
#         ### Article : Modèle mathématique pour la durabilité du tourisme dans les zones humides 
#         **Modélisation Mathématique** : Dr. Oumar Diop (oumar.diop@unchk.edu.sn) 
#         **Modélisation Mathématique** : Dr. Abdou Séne (abdou.sene@unchk.sn) 
#         """)
    
#     st.title("🌿 Simulateur de Dynamique des Zones Humides")
#     st.markdown(""" 
#     ### Exploration des interactions écologiques et touristiques 
#     Ce simulateur avancé modélise les interactions complexes entre : 
#     - Population de poissons 
#     - Population d'oiseaux 
#     - Afflux de touristes 
#     - Développement économique (capital)... 
#     Utilisez les paramètres à gauche pour explorer différents scénarios.
#     """)

#     simulator = WetlandSimulator()

#     with st.sidebar:
#         st.header("🛠 Paramètres de Simulation")
        
#         # Création des onglets pour organiser les paramètres
#         tab1, tab2, tab3, tab4 = st.tabs(["Écologie", "Tourisme", "Fonctions Périodiques", "Conditions Initiales"])
        
#         with tab1:
#             st.subheader("Paramètres Écologiques")
#             params = {
#                 'r': st.slider("Taux de croissance des poissons", min_value=0.1, max_value=1.0, value=simulator.default_params['r'], step=0.01),
#                 'K': st.slider("Capacité de charge des poissons", min_value=100, max_value=20000, value=int(simulator.default_params['K']), step=100),
#                 'alpha_B': st.slider("Impact de prédation", min_value=0.1, max_value=1.0, value=simulator.default_params['alpha_B'], step=0.01),
#                 'd': st.slider("Mortalité naturelle des oiseaux", min_value=0.05, max_value=1.0, value=simulator.default_params['d'], step=0.01),
#                 'beta': st.slider("Taux de reproduction des oiseaux", min_value=0.1, max_value=1.0, value=simulator.default_params['beta'], step=0.01)
#             }

#         with tab2:
#             st.subheader("Paramètres Touristiques")
#             params.update({
#                 'mu_B': st.slider("Attractivité des oiseaux", min_value=0.1, max_value=1.0, value=simulator.default_params['mu_B'], step=0.01),
#                 'mu_C': st.slider("Attractivité des infrastructures", min_value=0.1, max_value=1.0, value=simulator.default_params['mu_C'], step=0.01),
#                 'alpha_T': st.slider("Taux de pêche des touristes", min_value=0.0, max_value=1.0, value=simulator.default_params['alpha_T'], step=0.01)
#             })
        
#         with tab3:
#             st.subheader("Fonctions Périodiques")
#             params.update({
#                 'Lambda_Xm': st.slider("Amplitude de la migration des oiseaux", min_value=0.0, max_value=10.0, value=simulator.default_params['Lambda_Xm'], step=0.5),
#                 'a_Xm': st.slider("Amplitude de la variation de compétition", min_value=0.0, max_value=1.0, value=simulator.default_params['a_Xm'], step=0.01),
#                 'Gamma_Xm': st.slider("Amplitude de la pêche touristique", min_value=0.0, max_value=5.0, value=simulator.default_params['Gamma_Xm'], step=0.5)
#             })
        
#         with tab4:
#             st.subheader("Conditions Initiales")
#             initial_conditions = {
#                 'F_0': st.number_input("Population initiale de poissons", min_value=10, max_value=10000, value=simulator.default_initial_conditions['F_0']),
#                 'B_0': st.number_input("Population initiale d'oiseaux", min_value=10, max_value=500, value=simulator.default_initial_conditions['B_0']),
#                 'T_0': st.number_input("Nombre initial de touristes", min_value=10, max_value=200, value=simulator.default_initial_conditions['T_0']),
#                 'C_0': st.number_input("Capital initial", min_value=100, max_value=50000, value=simulator.default_initial_conditions['C_0'])
#             }
        
#         t_span = st.slider("Durée de simulation (jours)", min_value=100, max_value=3650, value=365)

#         # Mise à jour des paramètres restants
#         params.update({k: v for k, v in simulator.default_params.items() if k not in params})

#     results = simulator.simulate(params, initial_conditions, t_span)

#     if results is not None:
#         fig = create_plotly_figures(results)
#         st.plotly_chart(fig, use_container_width=True)

#         col1, col2 = st.columns(2)
#         with col1:
#             st.metric("Population finale de poissons", f"{results['Fish'].iloc[-1]:,.0f}")
#             st.metric("Population finale d'oiseaux", f"{results['Birds'].iloc[-1]:,.0f}")
#         with col2:
#             st.metric("Nombre final de touristes", f"{results['Tourists'].iloc[-1]:,.0f}")
#             st.metric("Capital final", f"{results['Capital'].iloc[-1]:,.0f}")

# if __name__ == "__main__":
#     main()
