import pandas as pd
from datetime import datetime
import urllib.request
import os

def load_omni_data(years):
    omni_data = {}
    for year in years:
        url = f"https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
        try:
            print(f"Baixando dados OMNI (clima espacial) para o ano {year}...")
            response = urllib.request.urlopen(url)
            for line in response:
                line_str = line.decode('utf-8')
                parts = line_str.split()
                if len(parts) > 50:
                    y = int(parts[0])
                    doy = int(parts[1])
                    hour = int(parts[2])
                    kp_raw = int(parts[38])      # WORD 39: Kp*10
                    dst = int(parts[40])          # WORD 41: Dst (nT)
                    bz_gsm = float(parts[16])     # WORD 17: Bz GSM (nT)
                    f107 = float(parts[50])        # WORD 51: F10.7 (sfu)
                    
                    # O valor de Kp vem multiplicado por 10 (ex: 23 = 2.3). 99 indica dado ausente.
                    kp = kp_raw / 10.0 if kp_raw != 99 else None
                    
                    # O valor de Dst tem ausente como 99999
                    if dst == 99999:
                        dst = None
                    
                    # Bz GSM: fill value = 999.9
                    if bz_gsm == 999.9:
                        bz_gsm = None
                    
                    # F10.7: fill value = 999.9
                    if f107 == 999.9:
                        f107 = None
                        
                    omni_data[(y, doy, hour)] = (kp, dst, bz_gsm, f107)
        except Exception as e:
            print(f"Erro ao baixar dados para {year}: {e}")
    return omni_data

def main():
    # Detecta se estamos rodando na raiz do projeto ou dentro de util/
    csv_path = 'data/epb_labels.csv'
    if not os.path.exists(csv_path):
        csv_path = '../data/epb_labels.csv'
        if not os.path.exists(csv_path):
            print("Erro: Não foi possível encontrar o arquivo epb_labels.csv. Certifique-se de executar no local correto.")
            return

    print(f"Lendo dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    
    sites = []
    datetimes = []
    filters = []
    years_to_fetch = set()
    
    print("Processando nomes de arquivos...")
    for filename in df['filename']:
        # Formato esperado: O6_CA_20251022_214347.png
        parts = filename.replace('.png', '').split('_')
        if len(parts) >= 4:
            filt = "630.0nm" if parts[0] == "O6" else parts[0]
            site = parts[1]
            date_str = parts[2]
            time_str = parts[3]
            
            try:
                dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                years_to_fetch.add(dt.year)
                datetimes.append(dt)
            except ValueError:
                datetimes.append(None)
                
            filters.append(filt)
            sites.append(site)
        else:
            filters.append(None)
            sites.append(None)
            datetimes.append(None)
            
    # Adicionando as colunas extras
    df['site'] = sites
    df['datetime'] = datetimes
    df['filter'] = filters
    
    if not years_to_fetch:
        print("Nenhuma data válida encontrada nos arquivos.")
        return

    omni_data = load_omni_data(years_to_fetch)
    
    print("Mapeando índices Kp, Dst, Bz GSM e F10.7...")
    kps = []
    dsts = []
    bzs = []
    f107s = []
    for dt in df['datetime']:
        if pd.notnull(dt):
            doy = dt.timetuple().tm_yday
            hour = dt.hour
            kp, dst, bz_gsm, f107 = omni_data.get((dt.year, doy, hour), (None, None, None, None))
            kps.append(kp)
            dsts.append(dst)
            bzs.append(bz_gsm)
            f107s.append(f107)
        else:
            kps.append(None)
            dsts.append(None)
            bzs.append(None)
            f107s.append(None)
            
    df['kp'] = kps
    df['dst'] = dsts
    df['bz_gsm'] = bzs
    df['f107'] = f107s
    
    print("Salvando arquivo atualizado...")
    df.to_csv(csv_path, index=False)
    
    print(f"\nSucesso! O dataset foi atualizado em {csv_path}.")
    print("Novas colunas:", list(df.columns))
    print("\nAmostra dos dados:")
    print(df[['filename', 'site', 'datetime', 'kp', 'dst', 'bz_gsm', 'f107']].head())

if __name__ == "__main__":
    main()
