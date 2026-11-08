#!/bin/bash

# Regenera clean/ (PNGs sem texto, para o modelo) de todas as noites já baixadas.
# Uso: ./regen_clean.sh [PASTA_BASE]
#
# Para cada PASTA_BASE/*/original/ chama `embrace.sh --offline --clean-only`.
# processed/ e os timelapses não são alterados. Se uma noite falhar, segue para
# as próximas e lista as falhas no final.
#
# No final imprime, por noite, as contagens de TIFs, clean/ e processed/, e sai
# com código 1 se alguma noite tiver menos arquivos em clean/ do que TIFs.
#
# PASTA_BASE é opcional (padrão: a pasta deste script, util/).

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EMBRACE="$SCRIPT_DIR/embrace.sh"
BASE_DIR=$(cd "${1:-$SCRIPT_DIR}" 2>/dev/null && pwd) || {
    echo -e "${RED}Erro: pasta base não encontrada: $1${NC}"
    exit 1
}

if [ ! -f "$EMBRACE" ]; then
    echo -e "${RED}Erro: embrace.sh não encontrado em $SCRIPT_DIR${NC}"
    exit 1
fi

shopt -s nullglob
NOITES=()
for orig in "$BASE_DIR"/*/original/; do
    NOITES+=("$(dirname "$orig")")
done

if [ ${#NOITES[@]} -eq 0 ]; then
    echo -e "${RED}Erro: nenhuma pasta */original/ em $BASE_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}=== Regenerando clean/ de ${#NOITES[@]} noite(s) em $BASE_DIR ===${NC}"

contar() {
    # Conta arquivos (não diretórios) com a extensão dada, sem diferenciar maiúsculas
    local dir="$1" ext="$2"
    [ -d "$dir" ] || { echo 0; return; }
    find "$dir" -maxdepth 1 -type f -iname "*.$ext" | wc -l
}

FALHAS=()
for noite in "${NOITES[@]}"; do
    nome=$(basename "$noite")
    echo -e "\n${GREEN}--- $nome ---${NC}"
    # stdin de /dev/null: embrace.sh não deve esperar entrada do terminal
    if ! bash "$EMBRACE" --offline --clean-only "$noite" </dev/null; then
        echo -e "${RED}Falha ao processar $nome${NC}"
        FALHAS+=("$nome (embrace.sh saiu com erro)")
    fi
done

echo -e "\n${GREEN}=== Resumo ===${NC}"
INCOMPLETAS=0
printf "%-16s %6s %6s %10s  %s\n" "noite" "tif" "clean" "processed" "status"
for noite in "${NOITES[@]}"; do
    nome=$(basename "$noite")
    n_tif=$(contar "$noite/original" tif)
    n_clean=$(contar "$noite/clean" png)
    n_proc=$(contar "$noite/processed" png)
    if [ "$n_clean" -lt "$n_tif" ]; then
        status="INCOMPLETO"
        INCOMPLETAS=$((INCOMPLETAS + 1))
        FALHAS+=("$nome (clean/ $n_clean < $n_tif TIFs)")
    else
        status="ok"
    fi
    printf "%-16s %6d %6d %10d  %s\n" "$nome" "$n_tif" "$n_clean" "$n_proc" "$status"
done

if [ ${#FALHAS[@]} -gt 0 ]; then
    echo -e "\n${RED}Falhas:${NC}"
    for f in "${FALHAS[@]}"; do
        echo -e "  - ${YELLOW}$f${NC}"
    done
fi

# A contagem decide o código de saída: embrace.sh sai com 0 mesmo
# quando quadros individuais falham.
[ $INCOMPLETAS -gt 0 ] && exit 1
exit 0
