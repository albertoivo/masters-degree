#!/bin/bash

# Script para baixar e processar arquivos .tif do INPE (EMBRACE)
# Uso: ./embrace.sh [--offline] [--clean-only] <URL_DO_DIRETORIO | PASTA_DA_NOITE>
#
# Opções (antes da URL/pasta):
#   --offline     não lista nem baixa nada; processa os TIFs já presentes em original/.
#                 Aceita a URL ou a pasta da noite (ex.: CA_20251022 ou util/CA_20251022).
#   --clean-only  gera só clean/; não toca em processed/ e não cria o timelapse.
#
# Saídas em <OBS>_<YYYYMMDD>/ (data de INÍCIO da noite; os arquivos usam a data UTC):
#   original/   TIFs 16 bits baixados
#   clean/      PNGs para o MODELO: mesmo realce e geometria, SEM texto nos pixels
#   processed/  PNGs anotados (estação, filtro, hora, data) para figuras e timelapse
#
# Os pixels de clean/ e processed/ são idênticos fora das áreas de texto.
# Se os TIFs já existirem em original/, não são baixados de novo; se a listagem
# da URL falhar, o script reprocessa os TIFs locais (nunca apaga uma pasta existente).

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

uso() {
    echo "Uso: $0 [--offline] [--clean-only] <URL_DO_DIRETORIO | PASTA_DA_NOITE>"
    echo "Exemplo: $0 https://embracedata.inpe.br/imager/CA/2025/CA_2025_1022/"
    echo "Exemplo: $0 --offline --clean-only CA_20251022"
}

OFFLINE=0
CLEAN_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --offline)    OFFLINE=1; shift ;;
        --clean-only) CLEAN_ONLY=1; shift ;;
        --)           shift; break ;;
        -*)           echo -e "${RED}Erro: opção desconhecida: $1${NC}"; uso; exit 1 ;;
        *)            break ;;
    esac
done

if [ -z "$1" ]; then
    echo -e "${RED}Erro: URL não fornecida${NC}"
    uso
    exit 1
fi
if [ $# -gt 1 ]; then
    echo -e "${RED}Erro: argumentos demais (as opções vêm antes da URL/pasta)${NC}"
    uso
    exit 1
fi

ALVO="${1%/}"
URL="${ALVO}/"
START_DIR=$(pwd)

if [ $OFFLINE -eq 1 ]; then
    echo -e "${GREEN}=== Iniciando processamento offline (sem download) ===${NC}"
else
    echo -e "${GREEN}=== Iniciando download e processamento ===${NC}"
fi
[ $CLEAN_ONLY -eq 1 ] && echo -e "${YELLOW}Modo --clean-only: só clean/; processed/ e timelapse não são alterados${NC}"
echo -e "Alvo: ${YELLOW}$URL${NC}"
echo ""

# Formato esperado: .../CA/2025/CA_2025_1022/
OBS_DATA=$(echo "$URL" | grep -oP '[A-Z]+_\d{4}_\d{4}' | tail -1)
if [ -n "$OBS_DATA" ]; then
    URL_OBS=$(echo "$OBS_DATA" | cut -d'_' -f1)
    URL_ANO=$(echo "$OBS_DATA" | cut -d'_' -f2)
    URL_MES_DIA=$(echo "$OBS_DATA" | cut -d'_' -f3)
    DIR_NAME="${URL_OBS}_${URL_ANO}${URL_MES_DIA}"
elif [ $OFFLINE -eq 1 ] && [[ $(basename "$ALVO") =~ ^[A-Z]+_[0-9]{8}$ ]]; then
    # Pasta da noite (OBS_YYYYMMDD), relativa ao diretório atual ou absoluta
    DIR_NAME="$ALVO"
else
    echo -e "${RED}Erro: Não foi possível extrair observatório e data da URL${NC}"
    echo "Formato esperado: .../OBSERVATORIO/YYYY/OBSERVATORIO_YYYY_MMDD/"
    [ $OFFLINE -eq 1 ] && echo "ou (com --offline) a pasta da noite: OBSERVATORIO_YYYYMMDD"
    exit 1
fi

DIR_EXISTED=0
[ -d "$DIR_NAME" ] && DIR_EXISTED=1
if [ $OFFLINE -eq 1 ] && [ ! -d "$DIR_NAME/original" ]; then
    echo -e "${RED}Erro: --offline requer TIFs em $DIR_NAME/original/, que não existe${NC}"
    exit 1
fi
mkdir -p "$DIR_NAME/original" "$DIR_NAME/clean"
[ $CLEAN_ONLY -eq 0 ] && mkdir -p "$DIR_NAME/processed"

echo -e "${GREEN}Diretório: ${YELLOW}$DIR_NAME${NC}"
echo -e "  - ${YELLOW}$DIR_NAME/original${NC}  (arquivos .tif)"
echo -e "  - ${YELLOW}$DIR_NAME/clean${NC}     (.png sem texto, para o modelo)"
[ $CLEAN_ONLY -eq 0 ] && echo -e "  - ${YELLOW}$DIR_NAME/processed${NC} (.png anotado, para figuras/vídeo)"
echo ""

cd "$DIR_NAME/original" || exit 1

# Extrai FILTRO, OBSERVATORIO, ANO, MES, DIA, DATA_FORMATADA e HORA_FORMATADA
# de um nome no formato O6_CA_YYYYMMDD_HHMMSS.<ext>. Retorna 1 se o nome não seguir o padrão.
extrair_info() {
    local basename="${1%.*}"
    local PARTS
    IFS='_' read -ra PARTS <<< "$basename"
    [ ${#PARTS[@]} -eq 4 ] || return 1
    [[ ${PARTS[2]} =~ ^[0-9]{8}$ ]] || return 1
    [[ ${PARTS[3]} =~ ^[0-9]{6}$ ]] || return 1

    FILTRO="${PARTS[0]}"
    OBSERVATORIO="${PARTS[1]}"
    ANO="${PARTS[2]:0:4}"; MES="${PARTS[2]:4:2}"; DIA="${PARTS[2]:6:2}"
    DATA_FORMATADA="$DIA/$MES/$ANO"
    HORA_FORMATADA="${PARTS[3]:0:2}:${PARTS[3]:2:2}:${PARTS[3]:4:2} UT"
    return 0
}

# Realce comum a todas as estações (inalterado em relação à versão anterior)
REALCE=(-contrast-stretch 0.5%x0.5% -gamma 0.7 -clahe 25x25%+256+2.5 -level 2%,99%)

# Geometria por estação: deixa as bolhas alinhadas na vertical (norte para cima)
geometria() {
    case "$1" in
        CA|BJL) GEOM=(-flop -rotate 90) ;;
        CP)     GEOM=(-flop) ;;
        STR)    GEOM=(-flip) ;;
        *)      return 1 ;;
    esac
    return 0
}

FALHA_DOWNLOAD=0
if [ $OFFLINE -eq 1 ]; then
    echo -e "${GREEN}[1/5] Listagem ignorada (--offline)${NC}"
    ARQUIVOS=$(ls O6_*.tif 2>/dev/null)
    if [ -z "$ARQUIVOS" ]; then
        echo -e "${RED}Nenhum arquivo O6_*.tif em $DIR_NAME/original/${NC}"
        exit 1
    fi
    echo -e "${GREEN}Encontrados $(echo "$ARQUIVOS" | wc -l) TIF(s) locais${NC}"
    echo -e "${GREEN}[2/5] Download ignorado (--offline)${NC}"
    echo ""
else
echo -e "${GREEN}[1/5] Buscando lista de arquivos...${NC}"
ARQUIVOS=$(curl -sf "$URL" | grep -o 'O6_[^"]*\.tif' | sort -u)

if [ -z "$ARQUIVOS" ]; then
    LOCAIS=$(ls O6_*.tif 2>/dev/null)
    if [ -n "$LOCAIS" ]; then
        echo -e "${YELLOW}Listagem da URL indisponível; reprocessando $(echo "$LOCAIS" | wc -l) TIF(s) locais${NC}"
        ARQUIVOS="$LOCAIS"
    else
        echo -e "${RED}Nenhum arquivo O6_*.tif encontrado na URL nem em original/${NC}"
        cd "$START_DIR" || exit 1
        # Só remove a pasta se ela foi criada agora (e está vazia)
        if [ $DIR_EXISTED -eq 0 ]; then
            rmdir "$DIR_NAME/original" "$DIR_NAME/clean" "$DIR_NAME/processed" "$DIR_NAME" 2>/dev/null
        fi
        exit 1
    fi
fi

TOTAL=$(echo "$ARQUIVOS" | wc -l)
echo -e "${GREEN}Encontrados $TOTAL arquivo(s)${NC}"
echo ""

echo -e "${GREEN}[2/5] Baixando arquivos...${NC}"
CONTADOR=1
for ARQUIVO in $ARQUIVOS; do
    if [ -s "$ARQUIVO" ]; then
        echo -e "${YELLOW}[$CONTADOR/$TOTAL]${NC} Já existe: $ARQUIVO"
    else
        echo -e "${YELLOW}[$CONTADOR/$TOTAL]${NC} Baixando: $ARQUIVO"
        # -f: erro HTTP não vira um .tif falso; arquivo parcial é removido
        if ! curl -sf -o "$ARQUIVO" "${URL}${ARQUIVO}"; then
            echo -e "   ${RED}✗ Falha no download${NC}"
            rm -f "$ARQUIVO"
            ((FALHA_DOWNLOAD++))
        fi
    fi
    ((CONTADOR++))
done
echo ""
fi

echo -e "${GREEN}[3/5] Processando arquivos com ImageMagick...${NC}"
CONTADOR=1
SUCESSO=0
FALHA=0
TOTAL_TIF=$(ls O6_*.tif 2>/dev/null | wc -l)
ERR_LOG=$(mktemp)

for TIF_FILE in O6_*.tif; do
    [ -f "$TIF_FILE" ] || continue
    BASE="${TIF_FILE%.tif}"
    CLEAN_FILE="../clean/${BASE}.png"
    PNG_FILE="../processed/${BASE}.png"
    echo -e "${YELLOW}[$CONTADOR/$TOTAL_TIF]${NC} Processando: $TIF_FILE"
    ((CONTADOR++))

    if ! extrair_info "$TIF_FILE"; then
        echo -e "   ${RED}✗ Nome fora do padrão O6_OBS_YYYYMMDD_HHMMSS.tif — ignorado${NC}"
        ((FALHA++)); continue
    fi
    if ! geometria "$OBSERVATORIO"; then
        echo -e "   ${RED}✗ Observatório desconhecido: $OBSERVATORIO — ignorado${NC}"
        ((FALHA++)); continue
    fi

    # 1) Versão limpa (modelo): realce + geometria, sem texto
    # 2) Versão anotada (figuras): mesmo comando + texto. O status verificado é o do magick.
    # clean/ sai em cinza 8 bits (como processed/): o PIL converte 16 bits para 'L' saturando, não reescalando
    if [ $CLEAN_ONLY -eq 1 ]; then
        if magick "$TIF_FILE" "${REALCE[@]}" "${GEOM[@]}" -colorspace Gray -depth 8 -quality 100 "$CLEAN_FILE" 2>"$ERR_LOG" \
           && [ -s "$CLEAN_FILE" ]; then
            echo -e "   ${GREEN}✓ Sucesso${NC}"
            ((SUCESSO++))
        else
            echo -e "   ${RED}✗ Erro no magick${NC}"
            cat "$ERR_LOG"
            ((FALHA++))
        fi
        continue
    fi

    if magick "$TIF_FILE" "${REALCE[@]}" "${GEOM[@]}" -colorspace Gray -depth 8 -quality 100 "$CLEAN_FILE" 2>"$ERR_LOG" \
       && magick "$TIF_FILE" "${REALCE[@]}" "${GEOM[@]}" \
            -font DejaVu-Sans \
            -gravity NorthWest -pointsize 24 -fill white -annotate +10+10 "$OBSERVATORIO" \
            -gravity NorthEast -pointsize 24 -fill white -annotate +10+10 "$FILTRO" \
            -gravity SouthWest -pointsize 24 -fill white -annotate +10+10 "$HORA_FORMATADA" \
            -gravity SouthEast -pointsize 24 -fill white -annotate +10+10 "$DATA_FORMATADA" \
            -quality 100 "$PNG_FILE" 2>>"$ERR_LOG" \
       && [ -s "$CLEAN_FILE" ] && [ -s "$PNG_FILE" ]; then
        echo -e "   ${GREEN}✓ Sucesso${NC}"
        ((SUCESSO++))
    else
        echo -e "   ${RED}✗ Erro no magick${NC}"
        cat "$ERR_LOG"
        ((FALHA++))
    fi
done
rm -f "$ERR_LOG"
echo ""

echo -e "${GREEN}[4/5] Processamento concluído!${NC}"
echo -e "Arquivos processados com sucesso: ${GREEN}$SUCESSO${NC}"
[ $FALHA -gt 0 ] && echo -e "Arquivos com erro: ${RED}$FALHA${NC}"
[ $FALHA_DOWNLOAD -gt 0 ] && echo -e "Downloads com erro: ${RED}$FALHA_DOWNLOAD${NC}"

cd "$START_DIR" || exit 1
echo -e "\nArquivos salvos em: ${YELLOW}$(cd "$DIR_NAME" && pwd)${NC}"
echo -e "TIF:       $(ls "$DIR_NAME/original/"*.tif 2>/dev/null | wc -l)"
echo -e "clean:     $(ls "$DIR_NAME/clean/"*.png 2>/dev/null | wc -l)"
echo -e "processed: $(ls "$DIR_NAME/processed/"*.png 2>/dev/null | wc -l)"
echo ""

# Timelapse (a partir das imagens anotadas)
if [ $CLEAN_ONLY -eq 1 ]; then
    echo -e "${GREEN}[5/5] Timelapse ignorado (--clean-only)${NC}"
elif [ $SUCESSO -gt 0 ]; then
    echo -e "${GREEN}[5/5] Gerando timelapse...${NC}"
    FIRST_PNG=$(ls "$DIR_NAME/processed/"*.png 2>/dev/null | head -1)
    if [ -n "$FIRST_PNG" ] && extrair_info "$(basename "$FIRST_PNG")"; then
        VIDEO_NAME="${FILTRO}_${OBSERVATORIO}_${ANO}${MES}${DIA}.mp4"
        VIDEO_PATH="$DIR_NAME/$VIDEO_NAME"
        echo -e "  Criando: ${YELLOW}$VIDEO_NAME${NC}"
        if (cd "$DIR_NAME/processed" && magick -delay 16 -loop 0 *.png "../$VIDEO_NAME") 2>/tmp/timelapse_error.log \
           && [ -f "$VIDEO_PATH" ]; then
            echo -e "  ${GREEN}✓ Timelapse criado: ${YELLOW}$VIDEO_PATH${NC}"
        else
            echo -e "  ${RED}✗ Erro ao criar timelapse${NC}"
            cat /tmp/timelapse_error.log
        fi
    fi
fi
