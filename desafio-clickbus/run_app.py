#!/usr/bin/env python
"""
Script para iniciar o aplicativo Streamlit do projeto ClickBus.
Este script contorna possíveis problemas de compatibilidade com diferentes versões do Streamlit.
"""

import os
import sys
import subprocess

def run_streamlit_app():
    """Executa o aplicativo Streamlit"""
    
    # Obter o caminho absoluto do diretório atual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Caminho para o aplicativo Streamlit
    app_path = os.path.join(current_dir, 'app.py')
    
    print(f"Iniciando aplicativo Streamlit: {app_path}")
    print("Para encerrar o aplicativo, pressione Ctrl+C")
    
    try:
        # Tentar executar com argumentos padrão
        result = subprocess.run(
            [sys.executable, "-m", "streamlit", "run", app_path],
            check=True
        )
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Erro ao iniciar o Streamlit: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nAplicativo encerrado pelo usuário.")
        return 0

if __name__ == "__main__":
    sys.exit(run_streamlit_app()) 