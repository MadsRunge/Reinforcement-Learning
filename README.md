# Reinforcement Learning - Taxi-v3 Q-Learning

Dette projekt implementerer Q-Learning algoritmen til at løse Taxi-v3 miljøet fra Gymnasium-biblioteket.

## Projektstruktur

```
Reinforcement-Learning/
├── q_learning_taxi.py      # Hovedfil med Q-Learning implementation
├── requirements.txt        # Python afhængigheder
├── README.md              # Denne fil
├── setup.sh              # Setup script (valgfri)
└── .python-version       # Specificerer Python version (auto-genereret)
```

## Forudsætninger

- Python 3.11.9 (eller nyere)
- pyenv installeret på dit system

### Installér pyenv (hvis ikke allerede installeret)

**macOS (via Homebrew):**
```bash
brew install pyenv
brew install pyenv-virtualenv
```

Tilføj til din shell konfiguration (`~/.zshrc` eller `~/.bash_profile`):
```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

## Setup og Installation

### Metode 1: Automatisk setup (Anbefalet)

Kør setup scriptet:
```bash
chmod +x setup.sh
./setup.sh
```

### Metode 2: Manuel setup

#### Trin 1: Installér Python 3.11.9

```bash
pyenv install 3.11.9
```

#### Trin 2: Opret virtuelt miljø

```bash
pyenv virtualenv 3.11.9 rl_taxi_env
```

#### Trin 3: Aktivér miljøet lokalt i projektet

```bash
pyenv local rl_taxi_env
```

Dette opretter en `.python-version` fil, som automatisk aktiverer miljøet når du er i projektmappen.

#### Trin 4: Installér afhængigheder

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Verificér Installation

Tjek at alt er installeret korrekt:
```bash
python --version  # Skulle vise Python 3.11.9
pip list          # Skulle vise gymnasium og numpy
```

## Brug

### Kør Q-Learning træning

```bash
python q_learning_taxi.py
```

Programmet vil:
- Træne en agent over 25,000 episodes
- Vise fremskridt hver 1000. episode
- Teste den trænede agent i 10 episodes
- Vise detaljerede statistikker

### Forventet output

```
Tilstandsrum størrelse: 500
Handlingsrum størrelse: 6
Q-tabel shape: (500, 6)

Starter træning...

Episode 1000/25000 - Gennemsnitlig reward (sidste 1000): -100.50 - Epsilon: 0.3660
Episode 2000/25000 - Gennemsnitlig reward (sidste 1000): -50.25 - Epsilon: 0.1340
...
Episode 25000/25000 - Gennemsnitlig reward (sidste 1000): 7.85 - Epsilon: 0.0100

Træning afsluttet!
```

## Hyperparametre

- **Learning Rate (α)**: 0.1
- **Discount Factor (γ)**: 0.6
- **Epsilon (ε)**: Starter ved 1.0, aftager til 0.01
- **Antal Episodes**: 25,000
- **Epsilon Decay**: 0.995 (eksponentiel aftagning)

## Om Taxi-v3 Miljøet

Taxi-v3 er et klassisk RL-problem hvor:
- En taxi skal samle en passager op på én lokation og aflevere på en anden
- **Tilstandsrum**: 500 diskrete tilstande
- **Handlingsrum**: 6 handlinger (syd, nord, øst, vest, pickup, dropoff)
- **Belønninger**:
  - +20 for korrekt aflevering
  - -1 per tidsstep
  - -10 for ulovlige pickup/dropoff

## Næste Skridt

For at udvide projektet kan du:
- Eksperimentere med forskellige hyperparametre
- Implementere andre RL-algoritmer (SARSA, DQN, etc.)
- Visualisere læringsprogressionen med matplotlib
- Gemme og indlæse trænede Q-tabeller
- Teste på andre Gymnasium miljøer

## Fejlfinding

### Miljøet aktiveres ikke automatisk
Sørg for at pyenv er korrekt konfigureret i din shell. Kør:
```bash
pyenv version
```

Du burde se `rl_taxi_env` i outputtet.

### ModuleNotFoundError
Sørg for at miljøet er aktiveret og kør:
```bash
pip install -r requirements.txt
```

### Python version konflikt
Hvis du får version konflikter, reinstallér miljøet:
```bash
pyenv virtualenv-delete rl_taxi_env
pyenv virtualenv 3.11.9 rl_taxi_env
pyenv local rl_taxi_env
pip install -r requirements.txt
```

## Deaktivering af Miljø

For at deaktivere det lokale miljø (fjern .python-version fil):
```bash
pyenv local --unset
```

Eller forlad blot projektmappen.

## Licens

Dette projekt er til uddannelsesformål.
