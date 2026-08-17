import pandas as pd
import ephem
from datetime import datetime
import os
import math

def get_site_coordinates(site_code):
    # Coordenadas aproximadas para cálculo da lua (Latitude, Longitude em string)
    sites = {
        'CA': ('-7.38', '-36.52'),   # Cariri
        'CP': ('-22.69', '-45.00'),  # Cachoeira Paulista
        'BJL': ('-13.25', '-43.41')  # Bom Jesus da Lapa
    }
    return sites.get(site_code, ('0.0', '0.0'))

def main():
    csv_path = 'data/epb_labels.csv'
    if not os.path.exists(csv_path):
        csv_path = '../data/epb_labels.csv'
        if not os.path.exists(csv_path):
            print("Erro: Não foi possível encontrar o arquivo epb_labels.csv.")
            return

    print(f"Lendo dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'datetime' not in df.columns or 'site' not in df.columns:
        print("Erro: O dataset precisa ter as colunas 'datetime' e 'site' geradas previamente.")
        return

    moon_illuminations = []
    moon_alts = []
    
    print("Calculando fase e elevação da Lua...")
    for index, row in df.iterrows():
        dt_str = row['datetime']
        site = row['site']
        
        if pd.isna(dt_str) or pd.isna(site):
            moon_illuminations.append(None)
            moon_alts.append(None)
            continue
            
        try:
            # Tenta converter a string de volta para datetime
            dt = datetime.strptime(str(dt_str), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            moon_illuminations.append(None)
            moon_alts.append(None)
            continue
            
        # Obter coordenadas
        lat, lon = get_site_coordinates(site)
        
        # Configurar observador (necessário para elevação e correções de fase local)
        obs = ephem.Observer()
        obs.lat = lat
        obs.lon = lon
        obs.date = dt # ephem aceita datetime do python diretamente (em UTC)
        
        # Calcular objeto Lua para a data e local
        moon = ephem.Moon(obs)
        
        # moon.phase = porcentagem de iluminação (0 a 100)
        # moon.alt = elevação no céu em radianos (convertido para graus)
        illumination = round(moon.phase, 2)
        altitude = round(math.degrees(moon.alt), 2)
        
        moon_illuminations.append(illumination)
        moon_alts.append(altitude)
        
    # Salvar as novas colunas
    df['moon_illumination'] = moon_illuminations
    df['moon_alt'] = moon_alts
    
    print("Salvando arquivo atualizado...")
    df.to_csv(csv_path, index=False)
    print(f"\nSucesso! O dataset foi atualizado em {csv_path} com os dados da Lua.")
    print("Amostra:")
    print(df[['filename', 'site', 'moon_illumination', 'moon_alt']].head())

if __name__ == "__main__":
    main()
