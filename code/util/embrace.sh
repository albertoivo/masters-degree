#!/bin/bash

# Script para baixar e processar arquivos .tif do INPE
# Uso: ./script.sh <URL_DO_DIRETORIO>

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verifica se a URL foi fornecida
if [ -z "$1" ]; then
    echo -e "${RED}Erro: URL não fornecida${NC}"
    echo "Uso: $0 <URL_DO_DIRETORIO>"
    echo "Exemplo: $0 https://embracedata.inpe.br/imager/CA/2025/CA_2025_1022/"
    exit 1
fi

URL="$1"

# Remove barra final se existir e adiciona novamente para padronizar
URL="${URL%/}/"

echo -e "${GREEN}=== Iniciando download e processamento ===${NC}"
echo -e "URL: ${YELLOW}$URL${NC}"
echo ""

# Extrai o observatório e a data da URL
# Formato esperado: .../CA/2025/CA_2025_1022/
OBS_DATA=$(echo "$URL" | grep -oP '[A-Z]+_\d{4}_\d{4}' | tail -1)

if [ -z "$OBS_DATA" ]; then
    echo -e "${RED}Erro: Não foi possível extrair observatório e data da URL${NC}"
    echo "Formato esperado: .../OBSERVATORIO/YYYY/OBSERVATORIO_YYYY_MMDD/"
    exit 1
fi

# Converte CA_2025_1022 para CA_20251022
OBSERVATORIO=$(echo "$OBS_DATA" | cut -d'_' -f1)
ANO=$(echo "$OBS_DATA" | cut -d'_' -f2)
MES_DIA=$(echo "$OBS_DATA" | cut -d'_' -f3)
DIR_NAME="${OBSERVATORIO}_${ANO}${MES_DIA}"

# Cria estrutura de diretórios
mkdir -p "$DIR_NAME/original"
mkdir -p "$DIR_NAME/processed"

echo -e "${GREEN}Diretório criado: ${YELLOW}$DIR_NAME${NC}"
echo -e "  - ${YELLOW}$DIR_NAME/original${NC} (arquivos .tif)"
echo -e "  - ${YELLOW}$DIR_NAME/processed${NC} (arquivos .png)"
echo ""

cd "$DIR_NAME/original"

# Função para extrair informações do nome do arquivo
extrair_info() {
    local filename="$1"
    local basename="${filename%.*}"  # Remove extensão
    
    # Divide por underscore
    IFS='_' read -ra PARTS <<< "$basename"
    
    # Verifica se tem 4 partes
    if [ ${#PARTS[@]} -ne 4 ]; then
        return 1
    fi
    
    FILTRO="${PARTS[0]}"
    OBSERVATORIO="${PARTS[1]}"
    DATA_STR="${PARTS[2]}"
    HORA_STR="${PARTS[3]}"
    
    # Converter data YYYYMMDD para DD/MM/YYYY
    if [[ ! $DATA_STR =~ ^[0-9]{8}$ ]]; then
        return 1
    fi
    
    ANO="${DATA_STR:0:4}"
    MES="${DATA_STR:4:2}"
    DIA="${DATA_STR:6:2}"
    DATA_FORMATADA="$DIA/$MES/$ANO"
    
    # Converter hora HHmmss para HH:mm:ss UT
    if [[ ! $HORA_STR =~ ^[0-9]{6}$ ]]; then
        return 1
    fi
    
    HORA="${HORA_STR:0:2}"
    MIN="${HORA_STR:2:2}"
    SEG="${HORA_STR:4:2}"
    HORA_FORMATADA="$HORA:$MIN:$SEG UT"
    
    return 0
}

echo -e "${GREEN}[1/4] Buscando lista de arquivos...${NC}"

# Baixa a página HTML e extrai os links dos arquivos O6_*.tif
ARQUIVOS=$(curl -s "$URL" | grep -o 'O6_[^"]*\.tif' | sort -u)

# Verifica se encontrou arquivos
if [ -z "$ARQUIVOS" ]; then
    echo -e "${RED}Nenhum arquivo O6_*.tif encontrado na URL${NC}"
    cd ../..
    rm -rf "$DIR_NAME"
    exit 1
fi

# Conta quantos arquivos foram encontrados
TOTAL=$(echo "$ARQUIVOS" | wc -l)
echo -e "${GREEN}Encontrados $TOTAL arquivo(s)${NC}"
echo ""

# Baixa cada arquivo
echo -e "${GREEN}[2/4] Baixando arquivos...${NC}"
CONTADOR=1
for ARQUIVO in $ARQUIVOS; do
    echo -e "${YELLOW}[$CONTADOR/$TOTAL]${NC} Baixando: $ARQUIVO"
    curl -s -O "${URL}${ARQUIVO}"
    ((CONTADOR++))
done
echo ""

# Processa cada arquivo com ImageMagick
echo -e "${GREEN}[3/4] Processando arquivos com ImageMagick...${NC}"

CONTADOR=1
SUCESSO=0
FALHA=0

for TIF_FILE in O6_*.tif; do
    if [ -f "$TIF_FILE" ]; then
        PNG_FILE="../processed/${TIF_FILE%.tif}.png"
        echo -e "${YELLOW}[$CONTADOR/$TOTAL]${NC} Processando: $TIF_FILE -> $(basename "$PNG_FILE")"
        
        # Extrai informações do nome do arquivo
        extrair_info "$TIF_FILE"
        
        # Processa com ImageMagick e adiciona anotações
        # Comando específico para cada observatório
        if [ "$OBSERVATORIO" = "CA" ] || [ "$OBSERVATORIO" = "BJL" ]; then
            magick "$TIF_FILE" \
                -contrast-stretch 0.5%x0.5% \
                -gamma 0.7 \
                -clahe 25x25%+256+2.5 \
                -level 2%,99% \
                -flop \
                -rotate 90 \
                -font DejaVu-Sans \
                -gravity NorthWest -pointsize 24 -fill white -annotate +10+10 "$OBSERVATORIO" \
                -gravity NorthEast -pointsize 24 -fill white -annotate +10+10 "$FILTRO" \
                -gravity SouthWest -pointsize 24 -fill white -annotate +10+10 "$HORA_FORMATADA" \
                -gravity SouthEast -pointsize 24 -fill white -annotate +10+10 "$DATA_FORMATADA" \
                -quality 100 \
                "$PNG_FILE" 2>&1 | tee /tmp/magick_error.log
        elif [ "$OBSERVATORIO" = "CP" ]; then
            magick "$TIF_FILE" \
                -contrast-stretch 0.5%x0.5% \
                -gamma 0.7 \
                -clahe 25x25%+256+2.5 \
                -level 2%,99% \
                -flop \
                -gravity NorthWest -pointsize 24 -fill white -annotate +10+10 "$OBSERVATORIO" \
                -gravity NorthEast -pointsize 24 -fill white -annotate +10+10 "$FILTRO" \
                -gravity SouthWest -pointsize 24 -fill white -annotate +10+10 "$HORA_FORMATADA" \
                -gravity SouthEast -pointsize 24 -fill white -annotate +10+10 "$DATA_FORMATADA" \
                -quality 100 \
                "$PNG_FILE" 2>&1 | tee /tmp/magick_error.log
        elif [ "$OBSERVATORIO" = "STR" ]; then
            magick "$TIF_FILE" \
                -contrast-stretch 0.5%x0.5% \
                -gamma 0.7 \
                -clahe 25x25%+256+2.5 \
                -level 2%,99% \
                -flip \
                -gravity NorthWest -pointsize 24 -fill white -annotate +10+10 "$OBSERVATORIO" \
                -gravity NorthEast -pointsize 24 -fill white -annotate +10+10 "$FILTRO" \
                -gravity SouthWest -pointsize 24 -fill white -annotate +10+10 "$HORA_FORMATADA" \
                -gravity SouthEast -pointsize 24 -fill white -annotate +10+10 "$DATA_FORMATADA" \
                -quality 100 \
                "$PNG_FILE" 2>&1 | tee /tmp/magick_error.log
        fi
        
        if [ $? -eq 0 ]; then
            if [ -f "$PNG_FILE" ]; then
                echo -e "   ${GREEN}✓ Sucesso${NC}"
                ((SUCESSO++))
            else
                echo -e "   ${RED}✗ Arquivo PNG não foi criado${NC}"
                cat /tmp/magick_error.log
                ((FALHA++))
            fi
        else
            echo -e "   ${RED}✗ Erro no comando magick${NC}"
            cat /tmp/magick_error.log
            ((FALHA++))
        fi
        
        ((CONTADOR++))
    fi
done
echo ""

# Resumo
echo -e "${GREEN}[4/5] Processamento concluído!${NC}"
echo -e "Arquivos processados com sucesso: ${GREEN}$SUCESSO${NC}"
if [ $FALHA -gt 0 ]; then
    echo -e "Arquivos com erro: ${RED}$FALHA${NC}"
fi

cd ../..
echo -e "\nArquivos salvos em: ${YELLOW}$(pwd)/$DIR_NAME${NC}"
echo -e "  - Arquivos TIF originais: ${YELLOW}$DIR_NAME/original/${NC}"
echo -e "  - Arquivos PNG processados: ${YELLOW}$DIR_NAME/processed/${NC}"
echo ""

# Lista os arquivos gerados
echo -e "${GREEN}Arquivos TIF baixados:${NC}"
ls -lh "$DIR_NAME/original/"*.tif 2>/dev/null | wc -l | xargs echo "Total:"
echo ""
echo -e "${GREEN}Arquivos PNG gerados:${NC}"
ls -lh "$DIR_NAME/processed/"*.png 2>/dev/null | wc -l | xargs echo "Total:"
echo ""

# Gerar timelapse
if [ $SUCESSO -gt 0 ]; then
    echo -e "${GREEN}[5/5] Gerando timelapse...${NC}"
    
    # Pegar o primeiro arquivo PNG para extrair informações
    FIRST_PNG=$(ls "$DIR_NAME/processed/"*.png 2>/dev/null | head -1)
    
    if [ -n "$FIRST_PNG" ]; then
        FIRST_BASENAME=$(basename "$FIRST_PNG")
        
        # Extrair informações do primeiro arquivo
        extrair_info "$FIRST_BASENAME"
        
        # Converter data de DD/MM/YYYY para YYYYMMDD
        DATA_VIDEO="${ANO}${MES}${DIA}"
        VIDEO_NAME="${FILTRO}_${OBSERVATORIO}_${DATA_VIDEO}.mp4"
        VIDEO_PATH="$DIR_NAME/$VIDEO_NAME"
        
        echo -e "  Criando: ${YELLOW}$VIDEO_NAME${NC}"
        
        # Gerar timelapse
        cd "$DIR_NAME/processed"
        if magick -delay 16 -loop 0 *.png "../$VIDEO_NAME" 2>&1 | tee /tmp/timelapse_error.log; then
            cd ../..
            if [ -f "$VIDEO_PATH" ]; then
                echo -e "  ${GREEN}✓ Timelapse criado com sucesso!${NC}"
                echo -e "  Localização: ${YELLOW}$VIDEO_PATH${NC}"
            else
                echo -e "  ${RED}✗ Arquivo de vídeo não foi criado${NC}"
                cat /tmp/timelapse_error.log
            fi
        else
            cd ../..
            echo -e "  ${RED}✗ Erro ao criar timelapse${NC}"
            cat /tmp/timelapse_error.log
        fi
    fi
fi