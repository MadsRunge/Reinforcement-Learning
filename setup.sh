#!/bin/bash

# Setup script for Reinforcement Learning Taxi-v3 projekt
# Dette script automatiserer setup af Python miljøet og afhængigheder

set -e  # Stop ved fejl

echo "==================================="
echo "Reinforcement Learning - Setup"
echo "==================================="
echo ""

# Farver til output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

PYTHON_VERSION="3.11.9"
VENV_NAME="rl_taxi_env"

# Tjek om pyenv er installeret
if ! command -v pyenv &> /dev/null; then
    echo -e "${RED}Fejl: pyenv er ikke installeret${NC}"
    echo "Installér pyenv først:"
    echo "  macOS: brew install pyenv pyenv-virtualenv"
    echo "  Linux: https://github.com/pyenv/pyenv#installation"
    exit 1
fi

echo -e "${GREEN}✓${NC} pyenv er installeret"

# Tjek om Python version er installeret
if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
    echo -e "${YELLOW}Python $PYTHON_VERSION er ikke installeret. Installerer nu...${NC}"
    pyenv install "$PYTHON_VERSION"
    echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION installeret"
else
    echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION er allerede installeret"
fi

# Tjek om virtual environment allerede eksisterer
if pyenv versions | grep -q "$VENV_NAME"; then
    echo -e "${YELLOW}Virtual environment '$VENV_NAME' eksisterer allerede.${NC}"
    read -p "Vil du slette og genoprette det? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Sletter eksisterende miljø..."
        pyenv virtualenv-delete -f "$VENV_NAME"
        echo -e "${GREEN}✓${NC} Gammelt miljø slettet"
    else
        echo "Springer oprettelse af miljø over..."
        pyenv local "$VENV_NAME"
        echo -e "${GREEN}✓${NC} Aktiveret eksisterende miljø"

        # Installér/opdater afhængigheder
        echo ""
        echo "Installerer/opdaterer afhængigheder..."
        pip install --upgrade pip
        pip install -r requirements.txt
        echo -e "${GREEN}✓${NC} Afhængigheder installeret"

        echo ""
        echo -e "${GREEN}Setup afsluttet!${NC}"
        echo ""
        echo "Næste skridt:"
        echo "  Kør programmet: python q_learning_taxi.py"
        exit 0
    fi
fi

# Opret virtual environment
echo "Opretter virtual environment '$VENV_NAME'..."
pyenv virtualenv "$PYTHON_VERSION" "$VENV_NAME"
echo -e "${GREEN}✓${NC} Virtual environment oprettet"

# Aktivér miljøet lokalt (opretter .python-version fil)
echo "Aktiverer miljøet lokalt i dette projekt..."
pyenv local "$VENV_NAME"
echo -e "${GREEN}✓${NC} Miljø aktiveret (via .python-version fil)"

# Opgradér pip
echo ""
echo "Opgraderer pip..."
pip install --upgrade pip
echo -e "${GREEN}✓${NC} pip opgraderet"

# Installér afhængigheder
echo ""
echo "Installerer afhængigheder fra requirements.txt..."
pip install -r requirements.txt
echo -e "${GREEN}✓${NC} Afhængigheder installeret"

# Verificér installation
echo ""
echo "==================================="
echo "Verificerer installation..."
echo "==================================="
echo ""

PYTHON_VER=$(python --version)
echo "Python version: $PYTHON_VER"

echo ""
echo "Installerede pakker:"
pip list | grep -E "gymnasium|numpy"

echo ""
echo -e "${GREEN}==================================="
echo "Setup afsluttet succesfuldt!"
echo -e "===================================${NC}"
echo ""
echo "Dit miljø er nu klar til brug!"
echo ""
echo "Næste skridt:"
echo "  1. Kør programmet: python q_learning_taxi.py"
echo ""
echo "Tips:"
echo "  - Miljøet aktiveres automatisk når du er i denne mappe"
echo "  - For at deaktivere: pyenv local --unset"
echo "  - For at se aktiv version: pyenv version"
echo ""
