#!/usr/bin/env python
"""
Script para instalar as dependências do projeto ClickBus.
Este script tenta diferentes métodos para instalar as dependências,
contornando possíveis problemas com versões específicas.
"""

import os
import sys
import subprocess
import platform
import time

def print_step(message):
    """Imprime uma mensagem de passo com formatação"""
    print("\n" + "=" * 60)
    print(f" {message}")
    print("=" * 60)

def run_command(command):
    """Executa um comando no terminal e retorna o código de saída"""
    print(f"Executando: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Erro (código {e.returncode}): {e}")
        return e.returncode

def install_dependencies():
    """Tenta instalar as dependências usando diferentes métodos"""
    
    print_step("Verificando ambiente Python")
    python_version = sys.version
    print(f"Python: {python_version}")
    print(f"Executável: {sys.executable}")
    print(f"Sistema: {platform.platform()}")
    
    # Verificar se pip está disponível
    print_step("Verificando pip")
    pip_check = run_command([sys.executable, "-m", "pip", "--version"])
    if pip_check != 0:
        print("Erro: pip não está disponível. Por favor, instale pip primeiro.")
        return 1
    
    # Atualizar pip
    print_step("Atualizando pip")
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Instalar setuptools
    print_step("Instalando setuptools")
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools", "wheel"])
    
    # Instalar dependências principais
    print_step("Instalando dependências principais individualmente")
    dependencies = [
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "scikit-learn>=1.4.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "streamlit>=1.32.0",
        "plotly>=5.18.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.29.0",
        "tqdm>=4.66.0",
    ]
    
    errors = 0
    for dep in dependencies:
        print(f"\nInstalando {dep}")
        result = run_command([sys.executable, "-m", "pip", "install", dep])
        if result != 0:
            print(f"Aviso: Falha ao instalar {dep}")
            errors += 1
            # Tentar instalar sem versão específica
            pkg_name = dep.split(">=")[0]
            print(f"Tentando instalar {pkg_name} sem versão específica...")
            run_command([sys.executable, "-m", "pip", "install", pkg_name])
            
    if errors > 0:
        print(f"\nAviso: {errors} dependências podem não ter sido instaladas corretamente.")
        print("O projeto ainda pode funcionar, mas pode haver problemas de compatibilidade.")
    else:
        print("\nTodas as dependências foram instaladas com sucesso!")
    
    print_step("Instalação concluída")
    print("Para executar o aplicativo, use:")
    print(f"   {sys.executable} run_app.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(install_dependencies()) 